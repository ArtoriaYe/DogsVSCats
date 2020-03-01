import os
import visdom
import torch
import torch.nn as nn
import numpy as np
from torch.optim.lr_scheduler import StepLR

from ignite.engine import Engine, Events
from ignite.handlers import Timer, Checkpoint, DiskSaver
from ignite.metrics import RunningAverage

import logging

from config import opt
from utils.metric_utils import CatDogAcc

logging.basicConfig(level=logging.INFO)


def global_step_from_engine(engine):
    """Helper method to setup `global_step_transform` function using another engine.
    This can be helpful for logging trainer epoch/iteration while output handler is attached to an evaluator.

    Args:
        engine (Engine): engine which state is used to provide the global step

    Returns:
        global step
    """

    def wrapper(_, event_name):
        return engine.state.get_event_attrib_value(Events.ITERATION_COMPLETED)

    return wrapper


def create_supervised_trainer(model, optimizer, loss_fn, device='cuda'):
    if device:
        if torch.cuda.device_count() > 1:
            model = nn.DataParallel(model)
        model.to(device)

    def _update(engine, batch):
        model.train()
        optimizer.zero_grad()
        img, label = batch
        img = img.to(device) if torch.cuda.device_count() >= 1 else img
        label = label.to(device).long() if torch.cuda.device_count() >= 1 else label
        out = model(img)
        loss = loss_fn(out, label)
        loss.backward()
        optimizer.step()
        return loss.item()

    return Engine(_update)


def create_supervised_evaluate(model, metrics, device='cuda'):
    if device:
        if torch.cuda.device_count() > 1:
            model = nn.DataParallel(model)
        model.to(device)

    def _inference(engine, batch):
        model.eval()
        with torch.no_grad():
            img, label = batch
            img = img.to(device) if torch.cuda.device_count() >= 1 else img
            label = label.to(device) if torch.cuda.device_count() >= 1 else label
            out = model(img)
            pre = torch.argmax(out, dim=1)
            return pre, label

    engine = Engine(_inference)

    for name, metric in metrics.items():
        metric.attach(engine, name)

    return engine


def do_train(model, train_loader, val_loader):
    global batch10loss
    batch10loss = 0

    viz = visdom.Visdom(env='test1')
    x = np.array(0)
    logger = logging.getLogger("DogVSCat_train")

    optimizer = torch.optim.SGD([{'params': model.parameters(), 'initial_lr': 0.001}], lr=0.001, momentum=0.9)
    lr_scheduler = StepLR(optimizer, step_size=1, gamma=0.5)
    loss_fn = nn.CrossEntropyLoss()

    trainer = create_supervised_trainer(model, optimizer, loss_fn)
    evaluator = create_supervised_evaluate(model, metrics={'acc': CatDogAcc()})

    def score_function(engine):
        acc = evaluator.state.metrics['acc']
        print("evaluator result is {:.3f}".format(acc))
        return evaluator.state.metrics['acc']

    object_to_checkpoint = {"trainer": trainer, "model": model, "optimizer": optimizer, "lr_scheduler": lr_scheduler}
    training_checkpoint = Checkpoint(
        to_save=object_to_checkpoint,
        score_function=score_function,
        global_step_transform=global_step_from_engine(trainer),
        score_name='val_acc',
        save_handler=DiskSaver(dirname=opt.checkpoint_dir, create_dir=True, require_empty=False, atomic=True, ),
        n_saved=2, filename_prefix='best'
    )

    timer = Timer(average=True)

    evaluator.add_event_handler(Events.COMPLETED, training_checkpoint)

    timer.attach(trainer, start=Events.EPOCH_STARTED, resume=Events.ITERATION_STARTED,
                 pause=Events.ITERATION_COMPLETED, step=Events.ITERATION_COMPLETED)

    RunningAverage(output_transform=lambda x: x).attach(trainer, 'avg_loss')

    if os.listdir(opt.checkpoint_dir):
        for root, dirs, files in os.walk(opt.checkpoint_dir):
            checkpoint_file = root + '//' + files[-1]
        checkpoint = torch.load(checkpoint_file)
        Checkpoint.load_objects(to_load=object_to_checkpoint, checkpoint=checkpoint)

    @trainer.on(Events.EPOCH_STARTED)
    def start_training(engine):
        logger.info("start {}th epoch training".format(engine.state.epoch))

    @trainer.on(Events.EPOCH_COMPLETED(every=2))
    def adjust_lr(engine):
        if engine.state.epoch >= 8:
            logger.info("last lr:{}".format(lr_scheduler.get_lr()[0]))
            lr_scheduler.step()
            logger.info("current lr:{}".format(lr_scheduler.get_lr()[0]))

    @trainer.on(Events.EPOCH_COMPLETED)
    def print_time(engine):
        logger.info('Epoch {} done. Time per batch:{:.3f}[s] Speed: {:.1f}[samples/s]'
                     .format(engine.state.epoch, timer.value() * timer.step_count,
                             train_loader.batch_size/timer.value()))
        logger.info('-' * 10)
        timer.reset()

    @trainer.on(Events.ITERATION_COMPLETED)
    def log_training(engine):
        batch_loss = engine.state.output
        lr = optimizer.param_groups[0]['lr']
        e = engine.state.epoch
        n = engine.state.max_epochs
        i = engine.state.iteration
        logger.info("Epoch {}/{} : {} - batch loss: {}, lr: {}".format(e, n, i, batch_loss, lr))
        x = np.array([i])
        global batch10loss
        batch10loss += batch_loss
        if i == 10:
            viz.line(X=x, Y=np.array([batch10loss / 100]), win='win')
            batch10loss = 0
        elif i % 10 == 0:
            viz.line(X=x, Y=np.array([batch10loss / 100]), update='append', win='win')
            batch10loss = 0

    @trainer.on(Events.ITERATION_COMPLETED(every=1))
    def evaluation(engine):
        evaluator.run(val_loader)
        acc = evaluator.state.metrics['acc']
        # print("evaluator result is {:.3f}".format(acc))
        i = engine.state.iteration
        x = np.array([i])
        if i == 2:
            viz.line(X=x, Y=np.array([acc]), win='evaluator\'s acc')
        else:
            viz.line(X=x, Y=np.array([acc]), win='evaluator\'s acc', update='append')

    trainer.run(train_loader, max_epochs=20)
