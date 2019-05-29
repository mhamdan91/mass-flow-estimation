from __future__ import absolute_import, division, print_function
from utils import functions
import warnings
import tensorflow as tf
import numpy as np
import time
warnings.filterwarnings("ignore")
tf.logging.set_verbosity(tf.logging.ERROR)  # disable to see tensorflow warnings
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
tf.enable_eager_execution(config=config)
tf.executing_eagerly()
print(tf.__version__)


# SETUP TENSORBOARD
def write_summaries(summary_writer, loss, i, global_step, vars_loc, grads_loc, train=True):
    with summary_writer.as_default():
        with tf.contrib.summary.always_record_summaries():
            if train:
                tf.contrib.summary.scalar("train_loss", loss, step=global_step)
                tf.contrib.summary.scalar("step", i, step=global_step)
                #  do not add spaces after names
                tf.contrib.summary.histogram("weights", vars_loc, step=global_step)
                tf.contrib.summary.histogram("gradients", grads_loc, step=global_step)
            else:
                tf.contrib.summary.scalar("val_loss", loss, step=global_step)


def validate_model(batch_size, model_loc, val_step_, n_logs, summary_writer, data_set, write_summary=True, return_losses=True):
    t = tf.constant(1 / 7.5, dtype="float32")
    logs_N = n_logs
    start = time.time()
    MAE = []
    MSE = []
    MSBE = []
    log_order = []
    Accuracy = []
    metric_mse = []
    summed_volume = 0
    remainder = 0
    iterations = 0
    fixed_size = 0
    count = 0
    new_log = True
    cnt_log = -1
    signal = []
    signals = []
    for (batch, (images, labels, speeds, log_length)) in (enumerate(data_set)):
        batch += 1
        if new_log:
            new_log = False
            length = int(log_length[0].numpy() - fixed_size + remainder)
            iterations = int(np.floor(length / batch_size))
            remainder = int(np.mod(length, batch_size))
            fixed_size = np.shape(speeds)[0]
            cnt_log += 1

        if count < iterations:
            mass_pred = model_loc(images)
            volume = (mass_pred * speeds) * t
            summed_volume += tf.reduce_sum(volume)
            mass = np.abs(summed_volume.numpy())
            count = count + 1
            signal.append(volume)
            # Compute loss
            if count == iterations and remainder == 0:
                loss_unsigned = np.squeeze(functions.compute_loss(summed_volume, labels[0], log_length[0], operation='L2').numpy())
                loss_outlier = np.squeeze(functions.compute_loss(summed_volume, labels[0], log_length[0], operation='L1').numpy())
                loss_signed = np.squeeze(functions.compute_loss(summed_volume, labels[0], log_length[0], operation='Subtraction').numpy())
                MSE.append(loss_unsigned)
                MAE.append(loss_outlier)
                MSBE.append(loss_signed)
                signals.append(signal[0:len(signal)])
                signal.clear()

                gt = np.squeeze(labels[0].numpy())
                metric_mse.append(loss_unsigned)

                # average total weight = 598*lb2kg = ~271 kg-- relative accuracy
                if gt > 0:
                    tmp = np.abs(gt - mass)
                    tmp = tmp / gt
                    log_acc = 1 - np.squeeze(tmp)
                else:
                    tmp = np.abs((271 + gt) - (271 + mass))
                    tmp = tmp / (271 + gt)
                    log_acc = 1 - np.squeeze(tmp)

                Accuracy.append(log_acc)
                log_order.append(np.squeeze(log_length[0].numpy()))
                time_ = time.time() - start
                functions.validation_progress(cnt_log, logs_N, time_, np.mean(metric_mse), np.mean(Accuracy))
                if write_summary:
                    write_summaries(summary_writer, np.mean(MSE), 0, val_step_, 0, 0, train=False)
                val_step_ += 1
                # Reset
                summed_volume = 0
                count = 0
                fixed_size = 0
                new_log = True
        # Handle the remainder from iterations
        else:
            # Compute and Aggregate the remainder Gradients
            mass_pred = model_loc(images[0:remainder])
            volume = (mass_pred * speeds[0:remainder]) * t
            summed_volume += tf.reduce_sum(volume)
            mass = np.abs(summed_volume.numpy())
            signal.append(volume)

            loss_unsigned = np.squeeze(functions.compute_loss(summed_volume, labels[0], log_length[0], operation='L2').numpy())
            loss_outlier = np.squeeze(functions.compute_loss(summed_volume, labels[0], log_length[0], operation='L1').numpy())
            loss_signed = np.squeeze(functions.compute_loss(summed_volume, labels[0], log_length[0], operation='Subtraction').numpy())
            MSE.append(loss_unsigned)
            MAE.append(loss_outlier)
            MSBE.append(loss_signed)
            signals.append(signal[0:len(signal)])
            signal.clear()

            gt = np.squeeze(labels[0].numpy())
            metric_mse.append(loss_unsigned)
            # average total weight = 598*lb2kg = ~271 kg-- relative accuracy
            if gt > 0:
                tmp = np.abs(gt - mass)
                tmp = tmp / gt
                log_acc = 1 - np.squeeze(tmp)
            else:
                tmp = np.abs((271+gt) - (271+mass))
                tmp = tmp / (271+gt)
                log_acc = 1 - np.squeeze(tmp)

            Accuracy.append(log_acc)
            log_order.append(np.squeeze(log_length[0].numpy()))
            time_ = time.time() - start
            functions.validation_progress(cnt_log, logs_N, time_, np.mean(metric_mse), np.mean(Accuracy))

            if write_summary:
                write_summaries(summary_writer, np.mean(MSE), 0, val_step_, 0, 0, train=False)
            val_step_ += 1
            summed_volume = 0
            count = 0
            new_log = True

            # Handle gradients for the next log
            if cnt_log != logs_N - 1:
                mass_pred = model_loc(images[remainder:])
                volume = (mass_pred * speeds[remainder:]) * t
                summed_volume += tf.reduce_sum(volume)
                signal.append(volume)

    if return_losses:
        return MSE, MAE, MSBE, Accuracy, val_step_
    else:
        return signals
