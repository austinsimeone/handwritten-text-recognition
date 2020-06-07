"""
Provides options via the command line to perform project tasks.
* `--source`: dataset/model name (bentham, iam, rimes, saintgall, washington)
* `--arch`: network to be used (puigcerver, bluche, flor)
* `--transform`: transform dataset to the HDF5 file
* `--cv2`: visualize sample from transformed dataset
* `--kaldi_assets`: save all assets for use with kaldi
* `--image`: predict a single image with the source parameter
* `--train`: train model with the source argument
* `--test`: evaluate and predict model with the source argument
* `--norm_accentuation`: discard accentuation marks in the evaluation
* `--norm_punctuation`: discard punctuation marks in the evaluation
* `--epochs`: number of epochs
* `--batch_size`: number of batches
"""

import argparse
import os
import string
import datetime

from data import DataLoader
from network.model import HTRModel


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=str, required=True)
    parser.add_argument("--path", type=str, required=True)
    parser.add_argument("--arch", type=str, default="flor")

    parser.add_argument("--cv2", action="store_true", default=False)
    parser.add_argument("--image", type=str, default="")
    parser.add_argument("--kaldi_assets", action="store_true", default=False)

    parser.add_argument("--train", action="store_true", default=False)
    parser.add_argument("--test", action="store_true", default=False)

    parser.add_argument("--norm_accentuation", action="store_true", default=False)
    parser.add_argument("--norm_punctuation", action="store_true", default=False)

    parser.add_argument("--epochs", type=int, default=1000)
    parser.add_argument("--batch_size", type=int, default=100)
    parser.add_argument("--image_size", type=list, default = [1024,128])
    parser.add_argument("--max_text_len", type=int,default = 32)
    args = parser.parse_args()

    source_path = os.path.join(args.path, args.source, 'words_screenshot_labeled')
    output_path = os.path.join(args.path, args.source,'output')
    target_path = os.path.join(args.path, args.source,'checkpoint')

    input_size = (1024, 128, 1)
    charset_base = string.printable[:95]

    ds = DataLoader.DataLoader(filePath = source_path,
                               batchSize = args.batch_size,
                               imgSize = args.image_size,
                               maxTextLen = args.max_text_len)

    model = HTRModel(architecture=args.arch,
                     input_size=input_size,
                     vocab_size=len(ds.charList))

    model.compile(learning_rate=0.001)
    model.load_checkpoint(target=target_path)

    if args.train:
        model.summary(output_path, "summary.txt")
        callbacks = model.get_callbacks(logdir=output_path, 
                                        checkpoint=target_path, 
                                        verbose=1)

        start_time = datetime.datetime.now()

        h = model.fit(x=ds.getNext().imgs,
                      epochs=args.epochs,
                      steps_per_epoch=ds.train_steps,
                      validation_data=ds.getNext(),
                      validation_steps=ds.valid_steps,
                      callbacks=callbacks,
                      shuffle=True,
                      verbose=1)

        total_time = datetime.datetime.now() - start_time

        loss = h.history['loss']
        val_loss = h.history['val_loss']

        min_val_loss = min(val_loss)
        min_val_loss_i = val_loss.index(min_val_loss)

        time_epoch = (total_time / len(loss))
        total_item = (len(ds.samples))

        t_corpus = "\n".join([
            f"Total train images:      {len(ds.partitions[0])}",
            f"Total validation images: {len(ds.partitions[1])}",
            f"Batch:                   {ds.batchSize}\n",
            f"Total time:              {total_time}",
            f"Time per epoch:          {time_epoch}",
            f"Time per item:           {time_epoch / total_item}\n",
            f"Total epochs:            {len(loss)}",
            f"Best epoch               {min_val_loss_i + 1}\n",
            f"Training loss:           {loss[min_val_loss_i]:.8f}",
            f"Validation loss:         {min_val_loss:.8f}"
        ])

        with open(os.path.join(output_path, "train.txt"), "w") as lg:
            lg.write(t_corpus)
            print(t_corpus)

    elif args.test:
        start_time = datetime.datetime.now()

        predicts, _ = model.predict(x=ds.getNext(),
                                    steps=ds.valid_steps,
                                    ctc_decode=True,
                                    verbose=1)

        predicts = [dtgen.tokenizer.decode(x[0]) for x in predicts]

        total_time = datetime.datetime.now() - start_time

        with open(os.path.join(output_path, "predict.txt"), "w") as lg:
            for pd, gt in zip(predicts, dtgen.dataset['test']['gt']):
                lg.write(f"TE_L {gt}\nTE_P {pd}\n")

        evaluate = evaluation.ocr_metrics(predicts=predicts,
                                          ground_truth=dtgen.dataset['test']['gt'],
                                          norm_accentuation=args.norm_accentuation,
                                          norm_punctuation=args.norm_punctuation)

        e_corpus = "\n".join([
            f"Total test images:    {dtgen.size['test']}",
            f"Total time:           {total_time}",
            f"Time per item:        {total_time / dtgen.size['test']}\n",
            f"Metrics:",
            f"Character Error Rate: {evaluate[0]:.8f}",
            f"Word Error Rate:      {evaluate[1]:.8f}",
            f"Sequence Error Rate:  {evaluate[2]:.8f}"
        ])

        sufix = ("_norm" if args.norm_accentuation or args.norm_punctuation else "") + \
                ("_accentuation" if args.norm_accentuation else "") + \
                ("_punctuation" if args.norm_punctuation else "")

        with open(os.path.join(output_path, f"evaluate{sufix}.txt"), "w") as lg:
            lg.write(e_corpus)
            print(e_corpus)
