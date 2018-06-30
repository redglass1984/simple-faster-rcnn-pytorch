import logging
import os
import ipdb
import matplotlib
from tqdm import tqdm

from utils.config import opt, init_logger
from data.dataset import Dataset, TestDataset, inverse_normalize
from model import FasterRCNNVGG16
from torch.autograd import Variable
from torch.utils import data as data_
from trainer import FasterRCNNTrainer
from utils import array_tool as at
from utils.vis_tool import tb_bbox
from utils.eval_tool import eval_detection_voc
from chainer import cuda
from tensorboardX import SummaryWriter
import json

matplotlib.use('agg')


def eval(dataloader, faster_rcnn, test_num=10000):
    pred_bboxes, pred_labels, pred_scores = list(), list(), list()
    gt_bboxes, gt_labels, gt_difficults = list(), list(), list()
    for ii, (imgs, sizes, gt_bboxes_, gt_labels_, gt_difficults_) in tqdm(enumerate(dataloader)):
        sizes = [sizes[0][0], sizes[1][0]]
        pred_bboxes_, pred_labels_, pred_scores_ = faster_rcnn.predict(imgs, [sizes])
        gt_bboxes += list(gt_bboxes_.numpy())
        gt_labels += list(gt_labels_.numpy())
        gt_difficults += list(gt_difficults_.numpy())
        pred_bboxes += pred_bboxes_
        pred_labels += pred_labels_
        pred_scores += pred_scores_
        if ii == test_num: break

    result = eval_detection_voc(
        pred_bboxes, pred_labels, pred_scores,
        gt_bboxes, gt_labels, gt_difficults,
        use_07_metric=True)
    return result


def train(**kwargs):
    opt._parse(kwargs)

    writer = SummaryWriter(opt.logdir)
    init_logger(opt.logdir)

    global_step = 0

    dataset = Dataset(opt)
    logging.info('load data')
    dataloader = data_.DataLoader(dataset,
                                  batch_size=1,
                                  shuffle=True,
                                  # pin_memory=True,
                                  num_workers=opt.num_workers)
    testset = TestDataset(opt)
    test_dataloader = data_.DataLoader(testset,
                                       batch_size=1,
                                       num_workers=opt.test_num_workers,
                                       shuffle=False,
                                       pin_memory=True
                                       )
    faster_rcnn = FasterRCNNVGG16()
    logging.info('model construct completed')
    trainer = FasterRCNNTrainer(faster_rcnn).cuda()
    if opt.load_path:
        trainer.load(opt.load_path)
        logging.info('load pretrained model from %s' % opt.load_path)

    logging.info(dataset.db.label_names)

    best_map = 0
    lr_ = opt.lr
    for epoch in range(opt.epoch):
        trainer.reset_meters()
        for ii, (img, bbox_, label_, scale) in tqdm(enumerate(dataloader)):
            global_step += 1
            if global_step % 100 == 99:
                break

            scale = at.scalar(scale)
            img, bbox, label = img.cuda().float(), bbox_.cuda(), label_.cuda()
            img, bbox, label = Variable(img), Variable(bbox), Variable(label)
            trainer.train_step(img, bbox, label, scale)

            if (ii) % opt.plot_every == 0:
                if os.path.exists(opt.debug_file):
                    ipdb.set_trace()

                # plot loss
                losses = trainer.get_meter_data()
                writer.add_scalars('losses', losses, global_step)
                logging.info('epoch {}, step {}: loss {}'.format(
                    epoch, ii, float(at.scalar(at.tonumpy(losses['total_loss'])))))

                # plot ground truth bboxes
                ori_img_ = inverse_normalize(at.tonumpy(img[0]))
                gt_img = tb_bbox(ori_img_,
                                 at.tonumpy(bbox_[0]),
                                 at.tonumpy(label_[0]))
                writer.add_image('gt_img', gt_img, global_step)

                # plot predicted bboxes
                _bboxes, _labels, _scores = trainer.faster_rcnn.predict([ori_img_], visualize=True)
                pred_img = tb_bbox(ori_img_,
                                   at.tonumpy(_bboxes[0]),
                                   at.tonumpy(_labels[0]).reshape(-1),
                                   at.tonumpy(_scores[0]))
                writer.add_image('pred_img', pred_img, global_step)

                # rpn confusion matrix(meter)
                writer.add_text('rpn_cm', str(trainer.rpn_cm.value().tolist()), global_step)
                # roi confusion matrix
                writer.add_image('roi_cm', at.totensor(trainer.roi_cm.conf, False).float(), global_step)

        # eval_result = eval(test_dataloader, faster_rcnn, test_num=opt.test_num)
        #
        # if eval_result['map'] > best_map:
        #     best_map = eval_result['map']
        #     best_path = trainer.save(best_map=best_map)
        # if epoch == 9:
        #     trainer.load(best_path)
        #     trainer.faster_rcnn.scale_lr(opt.lr_decay)
        #     lr_ = lr_ * opt.lr_decay
        #
        # writer.add_scalar('test_map', eval_result['map'], global_step)
        # log_info = 'lr:{}, map:{},loss:{}'.format(str(lr_), str(eval_result['map']), str(trainer.get_meter_data()))
        # logging.info('log', log_info, global_step)

    writer.close()


if __name__ == '__main__':
    import fire

    fire.Fire()
