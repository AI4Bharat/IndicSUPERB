# IndicSUPERB

IndicSUPERB is a robust benchmark consisting of 6 speech language understanding (SLU) tasks across 12 Indian languages. The tasks include automatic speech recognition, automatic speaker verification, speech idntification, query by example and keyword spotting. The IndicSUPERB also encompasses Kathbath dataset which has 1684 hours of labelled speech data across 12 Indian Languages.

Read more in our paper - [IndicSUPERB: A Speech Processing Universal Performance Benchmark for Indian languages](https://arxiv.org/pdf/2208.11761.pdf)

## Kathbath Dataset Details

|  | bengali| gujarati| hindi |kannada| malayalam| marathi| odia| punjabi| sanskrit| tamil| telugu| urdu| 
|-|-|-|-|-|-|-|-|-|-|-|-|-|
Data duration (hours) |115.8 |129.3 |150.2 |65.8 |147.3 |185.2 |111.6 |136.9 |115.5 |185.1 |154.9 |86.7|
No. of male speakers | 18 | 44 | 58 | 53 | 12 | 82 | 10 | 65 | 95 | 116 | 53 |36 |
No. of female speakers | 28	|35 |63	|26	|20	|61	|32	|77	|110 |42| 51|31|
Vocabulary (no. of unique words)|  6k | 109k  | 54k | 181k | 268k | 132k | 94k | 56k | 298k | 171k | 147k | 44k |


## Downloads
The dataset can be downloaded from the links given below.

#### Download Links (Clean split):
- Train: [85GB](https://indic-asr-public.objectstore.e2enetworks.net/indic-superb/kathbath/clean/train_audio.tar)
- Valid: [3GB](https://indic-asr-public.objectstore.e2enetworks.net/indic-superb/kathbath/clean/valid_audio.tar)
- Test Known: [2GB](https://indic-asr-public.objectstore.e2enetworks.net/indic-superb/kathbath/clean/testkn_audio.tar)
- Test Unknown: [2GB](https://indic-asr-public.objectstore.e2enetworks.net/indic-superb/kathbath/clean/testunk_audio.tar)

#### Transcripts: [clean](https://indic-asr-public.objectstore.e2enetworks.net/indic-superb/kathbath/clean/transcripts_n2w.tar)

#### Download Links (Noisy split):
- Test Known: [2GB](https://indic-asr-public.objectstore.e2enetworks.net/indic-superb/kathbath/noisy/testkn_audio.tar)
- Test Unknown: [1.4GB](https://indic-asr-public.objectstore.e2enetworks.net/indic-superb/kathbath/noisy/testunk_audio.tar)

#### Transcripts: [noisy](https://indic-asr-public.objectstore.e2enetworks.net/indic-superb/kathbath/noisy/transcripts_n2w.tar)

### Audio Dataset Format

- The audio files for each language are present in separate folders.
- The speaker and gender information are present in the name of the audio file.
- The audio files are stored in m4a format. For resampling, please check the sample code [here](https://github.com/AI4Bharat/IndicWav2Vec/tree/main/data_prep_scripts/ft_scripts)


#### Folder Structure of audios after extraction

```
Audio Data
data
├── bengali
│   ├── <split_name>
│   │   ├── 844424931537866-594-f.m4a
│   │   ├── 844424931029859-973-f.m4a
│   │   ├── ...
├── gujarati
├── ...


Transcripts
data
├── bengali
│   ├── <split_name>
│   │   ├── transcription_n2w.txt
├── gujarati
├── ...
```

## IndicSUPERB Tasks

In this section, we describe the tasks included in IndicSUPERB - automatic speech recognition, automatic speaker verification, speech idntification, query by example and keyword spotting.

### Download the Pretrained checkpoint

You can download the IndicWav2Vec pretrained checkpoint from [here](https://indic-asr-public.objectstore.e2enetworks.net/checkpoints/indicw2v/indicw2v_pretrained.pt).

### Data Preprocessing

To prepare data for the downstream tasks, the following 3 scripts need to be run to preprocess the raw downloaded data. You can download the data from [here](#downloads).

1. Convert m4a files to wav format
```
python utilities/structure.py \
  <dataset_root_path>/kb_data_clean_m4a \
  <dataset_root_path>/kb_data_clean_wav \
  <lang>
```

2. Preprocess data for ASR task
```
python utilities/preprocess_asr.py <dataset_root_path>/kb_data_clean_wav
```

### Language Identification

Language Identification, LID is the task of identifying the language of an audio clip. In other words, the task is to take the raw audio as input and classify it into one of the given n languages (n = 12 in this case).

#### Training - 
```
python s3prl/run_downstream.py -n lid_indicw2v -m train -u wav2vec2_local -d indic_lid -k <pretraining_checkpoint>
```

#### Evalutation -
```
overridebs="config.downstream_expert.datarc.eval_batch_size=1,,config.downstream_expert.datarc.file_path=<data_root>/kb_data_clean_wav"
python s3prl/run_downstream.py -m evaluate -e <trained_checkpoint> -t test_known -o $overridebs
python s3prl/run_downstream.py -m evaluate -e <trained_checkpoint> -t test_unknown -o $overridebs
```

### Speaker Identification

Speaker Identification, SID classifies each utterance for its speaker identity as a multi-class classification, where speakers are in the same predefined set for both training and testing.

#### Training -
```
python s3prl/run_downstream.py -n <exp_name> -m train -u wav2vec2_local -d indic_sid -o "config.downstream_expert.datarc.file_path=<data_root>/kb_data_clean_wav/<lang>" -k <pretraining_checkpoint>
```
#### Evaluation - 
```
overridebs="config.downstream_expert.datarc.eval_batch_size=1,,config.downstream_expert.datarc.file_path=<data_root>/kb_data_clean_wav/<lang>"
python s3prl/run_downstream.py -m evaluate -e <trained_checkpoint> -t test_known -o $overridebs
```

### Multilingual Speaker Identification

#### Training - 
```
python s3prl/run_downstream.py -n <exp_name> -m train -u wav2vec2_local -d indic_sid_multi -k <pretraining_checkpoint>
```
#### Evaluate - 
```
python s3prl/run_downstream.py -m evaluate -e <trained_checkpoint> -t test_known
python s3prl/run_downstream.py -m evaluate -e <trained_checkpoint> -t test_unknown
```

### Automatic Speaker Verification

Automatic Speaker Verification, ASV verifies whether the speakers of a pair of utterances match as a binary classification, and speakers in the testing set may not appear in the training set. Thus, ASV is more challenging than SID.

The meta data for ASV can be downloaded from [here](https://indic-asr-public.objectstore.e2enetworks.net/indic-superb/meta_data.zip).

#### Training - 
```
fp="<data_root>/kb_data_clean_wav/<lang>"
vmd="meta_data/<lang>/valid_data.txt"
tknmd="meta_data/$lang/test_known_data.txt"
tunkmd="meta_data/$lang/test_data.txt"
python s3prl/run_downstream.py \
  -n "asv_indicw2v_$lang" \
  -m train -u wav2vec2_local \
  -d indic_asv -k $pret_model\
  -o "config.downstream_expert.datarc.file_path=$fp,, \
  config.downstream_expert.datarc.valid_meta_data=$vmd,, \
  config.downstream_expert.datarc.test_meta_data=$tknmd,, \
  config.downstream_expert.datarc.test_unk_meta_data=$tunkmd,, \
  config.downstream_expert.datarc.lang=<lang>"
```
#### Evaluate -
```
l=[]
fp="<data_root>/kb_data_clean_wav/<lang>"
vmd="meta_data/<lang>/valid_data.txt"
tknmd="meta_data/$lang/test_known_data.txt"
tunkmd="meta_data/$lang/test_data.txt"
override="config.downstream_expert.datarc.eval_batch_size=1,,config.downstream_expert.datarc.file_path=$fp,,config.downstream_expert.datarc.valid_meta_data=$vmd,,config.downstream_expert.datarc.test_meta_data=$tknmd,,config.downstream_expert.datarc.test_unk_meta_data=$tunkmd,,config.downstream_expert.datarc.lang=$lang"
bash s3prl/downstream/indic_asv/test_expdir.sh <trained_checkpoint> $l $override
```

### Automatic Speech Recogntion

Automatic Speech Recognition, ASR is the task of transcribing a given audio utterance. The evaluation metric is word error rate (WER).

We use the [fairseq](https://github.com/facebookresearch/fairseq/tree/main/examples/wav2vec) library to train ASR models. We provide the data manifest files and the character dictionaries.

- Manifest files - `<data_root>/kb_data_clean_wav/<lang>/train/manifest`
- Dicts - `s3prl/downstream/indic_asr/dicts`

For detailed instructions on training ASR models, refer to [IndicWav2vec](https://github.com/AI4Bharat/IndicWav2Vec).

### Query By Example

Query by Example Spoken Term Detection, QbE detects a spoken term (query) in an audio database (documents) by binary discriminating a given pair of query and document into a match or not. The evaluation metric is maximum term weighted value (MTWV) which balances misses and false alarms.

Download the Query by Example dataset from [here](https://indic-asr-public.objectstore.e2enetworks.net/indic-superb/qbe_indicsuperb.zip)[3GB].

#### Evaluate -

Modify lines 1 and 2 in `superb/qbe_new/qbe_indicsuperb/SCORING/score_lang_agg.sh`.

```
bash qbe_indicsuperb/SCORING/score_lang_agg.sh
```


## Citing our work

If you are using any of the resources, please cite the following article:

```
@misc{https://doi.org/10.48550/arxiv.2208.11761,
  doi = {10.48550/ARXIV.2208.11761},
  url = {https://arxiv.org/abs/2208.11761},
  author = {Javed, Tahir and Bhogale, Kaushal Santosh and Raman, Abhigyan and Kunchukuttan, Anoop and Kumar, Pratyush and Khapra, Mitesh M.},
  title = {IndicSUPERB: A Speech Processing Universal Performance Benchmark for Indian languages},
  publisher = {arXiv},
  year = {2022},
  copyright = {arXiv.org perpetual, non-exclusive license}
}
```

We would like to hear from you if:
 - You are using our resources. Please let us know how you are putting these resources to use.
 - You have any feedback on these resources.

## License

### Dataset

The IndicSUPERB dataset is released under this licensing scheme:
- We do not own any of the raw text used in creating this dataset. The text data comes from the IndicCorp dataset which is a crawl of publicly available websites.
- The audio transcriptions  of the raw text and labelled annotations of the datasets have been created by us.
- We license the actual packaging of all this data under the Creative Commons CC0 license (“no rights reserved”).
- To the extent possible under law, AI4Bharat has waived all copyright and related or neighboring rights to the IndicSUPERB dataset.
- This work is published from: India.

### Code and Models
The IndicSUPERB code and models are released under the MIT License.

## Contributors
- Tahir Javed
- Kaushal Bhogale
- Abhigyan Raman
- Anoop Kunchukuttan
- Mitesh Khapra
- Pratush Kumar

## Contact
- Anoop Kunchukuttan (anoop.kunchukuttan@gmail.com)
- Mitesh Khapra (miteshk@cse.iitm.ac.in)
- Pratyush Kumar (pratyush@cse.iitm.ac.in)

## Acknowledgements

We would like to thank the Ministry of Electronics and Information Technology [(MeitY)](https://www.meity.gov.in/) of the Government of India and the Centre for Development of Advanced Computing [(C-DAC)](https://www.cdac.in/index.aspx?id=pune), Pune for generously supporting this work and providing us access to multiple GPU nodes on the Param Siddhi Supercomputer. We would like to thank the EkStep Foundation and Nilekani Philanthropies for their generous grant which went into hiring human resources as well as cloud resources needed for this work. We would like to thank DesiCrew for connecting us to native speakers for collecting data. We would like to thank Vivek Seshadri from Karya Inc. for helping setup the data collection infrastructure on the Karya platform. We would like to thank all the members of AI4Bharat team in helping create the Query by Example dataset. 





