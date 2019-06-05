from fairseq import data, options, tasks, utils

# Parse command-line arguments for generation
parser = options.get_generation_parser(default_task='len_pre_transformer')
args = options.parse_args_and_arch(parser)

# Setup task
task = tasks.setup_task(args)

# Load model
print('| loading model from {}'.format(args.path))
models, _model_args = utils.load_ensemble_for_inference([args.path], task)
model = models[0]

src_path = "/n/home05/simonx/scratchlfs/zhenyu/length_predic_transforemr/diff_len/data-process/test.de-en.de"
tgt_path = "/n/home05/simonx/scratchlfs/zhenyu/length_predic_transforemr/diff_len/data-process/test.de-en.en"
src_r = open(src_path,'r',encoding='UTF-8')
tgt_r = open(tgt_path,'r',encoding='UTF-8')

src_text = src_r.readline()
tgt_text = tgt_r.readline().strip()

total_num = 0
acc = 0
while total_num < 100:
    sentence = src_text

    # Tokenize into characters
    chars = ' '.join(list(sentence.strip()))
    tokens = task.source_dictionary.encode_line(
        chars, add_if_not_exist=False,
    )

    # Build mini-batch to feed to the model
    batch = data.language_pair_dataset.collate(
        samples=[{'id': -1, 'source': tokens}],  # bsz = 1
        pad_idx=task.source_dictionary.pad(),
        eos_idx=task.source_dictionary.eos(),
        left_pad_source=False,
        input_feeding=False,
    )

    # Feed batch to the model and get predictions
    preds = model(**batch['net_input'])

    # Print top 3 predictions and their log-probabilities
    top_scores, top_labels = preds[0].topk(k=5)
    for score, label_idx in zip(top_scores, top_labels):
        label_name = task.target_dictionary.string([label_idx])
        print('({:.2f})\t{} '.format(score, label_name))
    print("-----true length is {}".format(tgt_text))
    if (int(tgt_text)  in top_labels):
        print("successful")
        acc += 1
    total_num += 1
    src_text = src_r.readline()
    tgt_text = tgt_r.readline().strip()

src_r.close()
tgt_r.close()
print('--------------acc is -------------')
print(acc/total_num)

