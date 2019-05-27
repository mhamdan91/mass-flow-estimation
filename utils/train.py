from __future__ import absolute_import, division, print_function
from utils import functions
from utils.validate import validate_model
import warnings
import tensorflow as tf
import os
import numpy as np
import time
import pickle as pk
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


# TRAINING CODE  -- or move the following cell if you just wish to test results and make plots

def train(Epochs, batch_size, model, summary_writer, train_dataset, validation_dataset, MAIN_dir, checkpoint_path, path_sep, logs_N):
    optimizer = tf.train.AdamOptimizer()
    acc_prediction = tf.constant(0, dtype="float32")
    loss = tf.constant(0, dtype="float32")
    t = tf.constant(1 / 7.5, dtype="float32")
    loc_checkpoint_path = checkpoint_path + 'generated_checkpoints' + path_sep + "cp-{log:06d}.ckpt"
    checkpoint_dir = os.path.dirname(loc_checkpoint_path)
    save_epoch = 1
    init_vol = 0
    summed_vol_diff = 0
    volume_diff = []
    aggregated_diff = []
    lamda = 0.05
    # logs_N = 3
    start = time.time()
    aggregated = []
    MSE_log = []
    MAE_log = []
    MSBE_log = []
    MSE_t = []
    MAE_t = []
    MSBE_t = []
    MSE_avg = []
    MAE_avg = []
    MSBE_avg = []
    MSE_train = []
    val_step = 1
    run_name = 'test_run'
    performance_flag = True
    for epoch in range(Epochs):
        loss_metric = 0
        summed_volume = 0
        remainder = 0
        iterations = 0
        fixed_size = 0
        count = 0
        new_log = True
        append_flag = True
        next_log = False
        cnt_log = -1
        first_run = True
        for (batch, (images, labels, speeds, log_length)) in (enumerate(train_dataset)):
            batch += 1
            step = tf.train.get_or_create_global_step()
            if new_log:
                new_log = False
                length = int(log_length[0].numpy() - fixed_size + remainder)
                iterations = int(np.floor(length / batch_size))
                remainder = int(np.mod(length, batch_size))
                fixed_size = np.shape(speeds)[0]
                cnt_log += 1

            if count < iterations:

                # Compute and Aggregate Gradients
                volume_diff.clear()
                with tf.GradientTape(persistent=True) as tape:
                    mass_pred = model(images)
                    # print(np.shape(mass_pred))
                    volume = (mass_pred * speeds) * t
                    # print(volume)
                    for ixd, vol in enumerate(volume):
                        if ixd == 0:
                            summed_vol_diff += tf.squared_difference(volume[ixd], init_vol)
                            volume_diff.append(tf.squared_difference(volume[ixd], init_vol))
                        else:
                            summed_vol_diff += tf.squared_difference(volume[ixd], volume[ixd - 1])
                            volume_diff.append(tf.squared_difference(volume[ixd], volume[ixd - 1]))
                    init_vol = volume[batch_size - 1]
                    summed_volume += tf.reduce_sum(volume)
                    watched_vars = tape.watched_variables()
                grads = tape.gradient(volume, model.trainable_variables)
                grads_diff = tape.gradient(volume_diff, model.trainable_variables)
                del tape

                if count == 0 and append_flag:
                    for idx, grad in enumerate(grads):
                        aggregated.append(grad)
                        aggregated_diff.append(grads_diff[idx])
                else:
                    for idx, grad in enumerate(grads):
                        aggregated[idx] = grad + aggregated[idx]
                        aggregated_diff[idx] = grads_diff[idx] + aggregated_diff[idx]

                count = count + 1

                # Compute loss and Apply Gradients
                if count == iterations and remainder == 0:
                    loss = functions.compute_loss(summed_volume, labels[0], log_length[0], operation='Subtraction')
                    loss_metric = tf.squeeze(functions.compute_loss(summed_volume, labels[0], log_length[0], operation='L2').numpy())
                    for idx, grd in enumerate(aggregated):
                        aggregated[idx] = (loss * 2 * grd) + (summed_vol_diff * 2 * lamda * aggregated_diff[idx] / log_length[0])
                    rmse = tf.sqrt(tf.abs(loss_metric))
                    optimizer.apply_gradients(zip(aggregated, model.trainable_variables), global_step=step)
                    var_list = model.variables
                    write_summaries(summary_writer, loss_metric, batch, step, var_list[0], grads[0], train=True)
                    end = time.time() - start
                    sec = int(end % 60)
                    mint = int(end / 60) % 60
                    hr = int(end / 3600) % 60
                    # print(volume[0])
                    print("\r Time_lapsed (hr:mm:ss) --> {:02d}:{:02d}:{:02d} Epoch: {}, Log: {}, Log_progress: {:.1%} - Overall_progress: {:.1%}, Length: {} "
                          "Label:{}, loss: {:.3f}, train RMSE: {:.2f}".format(hr, mint, sec, epoch + 1, cnt_log + 1, (cnt_log + 1) / logs_N,
                                                                              (epoch + 1) / Epochs,
                                                                              log_length[0].numpy(), labels[0].numpy(), loss_metric, rmse))

                    if first_run:
                        first_run = False
                        loss_ = loss_metric.numpy()

                    # Reset
                    aggregated.clear()
                    aggregated_diff.clear()
                    summed_vol_diff = 0
                    init_vol = 0
                    summed_volume = 0
                    count = 0
                    append_flag = True
                    fixed_size = 0
                    new_log = True

            # Handle the remainder from iterations
            else:
                # Compute and Aggregate the remainder Gradients
                volume_diff.clear()
                with tf.GradientTape(persistent=True) as tape:
                    mass_pred = model(images[0:remainder])
                    volume = (mass_pred * speeds[0:remainder]) * t
                    for ixd, vol in enumerate(volume):
                        if ixd == 0:
                            summed_vol_diff += tf.squared_difference(volume[ixd], init_vol)
                            volume_diff.append(tf.squared_difference(volume[ixd], init_vol))
                        else:
                            summed_vol_diff += tf.squared_difference(volume[ixd], volume[ixd - 1])
                            volume_diff.append(tf.squared_difference(volume[ixd], volume[ixd - 1]))
                    init_vol = volume[remainder - 1]
                    summed_volume += tf.reduce_sum(volume)
                    watched_vars = tape.watched_variables()
                grads = tape.gradient(volume, model.trainable_variables)
                grads_diff = tape.gradient(volume_diff, model.trainable_variables)
                del tape

                for idx, grad in enumerate(grads):
                    aggregated[idx] = grad + aggregated[idx]
                    aggregated_diff[idx] = grads_diff[idx] + aggregated_diff[idx]

                    # Compute loss and apply gradients for the remainder
                loss = functions.compute_loss(summed_volume, labels[0], log_length[0], operation='Subtraction')
                loss_metric = tf.squeeze(functions.compute_loss(summed_volume, labels[0], log_length[0], operation='L2').numpy())
                for idx, grd in enumerate(aggregated):
                    aggregated[idx] = (loss * 2 * grd) + (summed_vol_diff * 2 * lamda * aggregated_diff[idx] / log_length[0])
                rmse = tf.squeeze(tf.sqrt(tf.abs(loss_metric)))
                optimizer.apply_gradients(zip(aggregated, model.trainable_variables), global_step=step)
                var_list = model.trainable_variables
                write_summaries(summary_writer, loss_metric, batch, step, var_list[0], grads[0], train=True)
                end = time.time() - start
                sec = int(end % 60)
                mint = int(end / 60) % 60
                hr = int(end / 3600) % 60
                # print(volume[0])
                print("\r Time_lapsed (hr:mm:ss) --> {:02d}:{:02d}:{:02d} Epoch: {}, Log: {}, Log_progress: {:.1%} - Overall_progress: {:.1%}, Length: {} "
                      "Label:{}, loss: {:.3f}, train RMSE: {:.2f}".format(hr, mint, sec, epoch + 1, cnt_log + 1, (cnt_log + 1) / logs_N, (epoch + 1) / Epochs,
                                                                          log_length[0].numpy(), labels[0].numpy(), loss_metric, rmse))

                if first_run:
                    first_run = False
                    loss_ = loss_metric.numpy()

                # Reset
                aggregated.clear()
                aggregated_diff.clear()
                summed_vol_diff = 0
                init_vol = 0
                summed_volume = 0
                count = 0
                append_flag = True
                new_log = True

                # Handle gradients for the next log
                if cnt_log != logs_N - 1:
                    # Compute and Aggregate Gradients
                    volume_diff.clear()
                    with tf.GradientTape(persistent=True) as tape:
                        mass_pred = model(images[remainder:])
                        volume = (mass_pred * speeds[remainder:]) * t
                        for ixd, vol in enumerate(volume):
                            if ixd == 0:
                                summed_vol_diff += tf.squared_difference(volume[ixd], init_vol)
                                volume_diff.append(tf.squared_difference(volume[ixd], init_vol))
                            else:
                                summed_vol_diff += tf.squared_difference(volume[ixd], volume[ixd - 1])
                                volume_diff.append(tf.squared_difference(volume[ixd], volume[ixd - 1]))
                        init_vol = volume[batch_size - remainder - 1]

                        summed_volume += tf.reduce_sum(volume)
                        watched_vars = tape.watched_variables()
                    grads = tape.gradient(volume, model.trainable_variables)
                    grads_diff = tape.gradient(volume_diff, model.trainable_variables)
                    del tape

                    for idx, grad in enumerate(grads):
                        aggregated.append(grad)
                        aggregated_diff.append(grads_diff[idx])
                    append_flag = False
        # Validate model every epoch to determine early stopping -- not necessary in this single log case
        MSE_log, MAE_log, MSBE_log, accuracy_, val_step = validate_model(batch_size, model, val_step, 1, summary_writer, validation_dataset, write_summary=True,
                                                                         return_losses=True)
        print('\n')
        MSE_t.append(MSE_log)  # contains log losses for each epoch
        MAE_t.append(MAE_log)
        MSBE_t.append(MSBE_log)
        MSE_avg.append(np.mean(MSE_log))  # contains average log losses for each epoch
        MAE_avg.append(np.mean(MAE_log))
        MSBE_avg.append(np.mean(MSBE_log))
        MSE_train.append(loss_metric)
        error = 1 - np.mean(accuracy_)
        full_data_dict = {"MSE_t": MSE_t, "MAE_t": MAE_t, "MSBE_t": MSBE_t, "MSE_avg": MSE_avg, "MAE_avg": MAE_avg, "MSBE_avg": MSBE_avg,
                          "MSE_train": MSE_train}
        pickle_out = open(MAIN_dir + "data_files/" + "losses_" + run_name + ".pickle", "wb")
        pk.dump(full_data_dict, pickle_out)
        pickle_out.close()

        # save model weights when performance flag for 94% and above
        if performance_flag:
            if error <= 0.06:
                model.save_weights(loc_checkpoint_path.format(log=epoch))
        else:
            if epoch >= save_epoch:
                model.save_weights(loc_checkpoint_path.format(log=epoch))

        if error <= 0.005:  # end training for 99.5% -- change this for higher or lower accuracy termination
            break

    model.save_weights(checkpoint_dir + '/cp-' + run_name + '.ckpt')
