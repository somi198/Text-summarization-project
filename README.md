# [NLP]  KoBART를 사용한 텍스트 요약 모델

#### 관련 연구 동향 및 논문 리서치

최근 텍스트 정보의 기하급수적인 증가로 인해 사용자는 원하는 정보를 찾는데 더욱 많은 시간을 투자하여 해당 정보를 일일히 확인하여 찾아야 하는 불편함이 가중되고 있다. 
이러한 문제를 해결하기 위해 정보 검색, 텍스트 마이닝을 이용한 문서 요약에 대한 관심이 고조되면서 이에 대한 연구개발 투자와 많은 연구들이 수행되고 있다. 또한 영상 및 텍스트 분야의 빠른 콘텐츠 소비 트렌드가 반영된 숏 폼 콘텐츠(Short-Form contents)의 수요가 급증하며 문서 요약의 중요도가 점점 높아지는 추세이다.
관련 기술로는 Text Rank, Attention mechanism, KoBART, koBERT, BERT with copy mechanism 등이 있다.


본프로젝트는 텍스트 요약에 특화된 성능을 띄는 `KoBART` 모델을 기반으로하여, 병원 전화상담센터에서 고객과의 통화 내용을 효율적으로 요약할 수 있는 모델을 만들고 성능을 개선하는 것을 목표로 두고 있다.

***

## 1. Data
- [Dacon 한국어 문서 생성요약 AI 경진대회](https://dacon.io/competitions/official/235673/overview/description)의 학습데이터를 활용함
- [AI hub의 문서요약텍스트](https://aihub.or.kr/aidata/8054) 데이터를 활용함
- [AI hub의 한국어대화요약](https://aihub.or.kr/aidata/30714) 데이터를 활용함
- 학습 데이터에서 임의로 Train/Test 데이터를 생성함
- 데이터 탐색에 용이하게 csv 형태로 데이터를 변환함
- Data 구조: **train set 크기 (test set 크기)**
  - Dacon 신문기사: 34,242 (8,501)
  - AI hub 문서요약텍스트: 321,052 (40,132)
     - 법률문서: 24,029 (3,004)
     - 신문기사: 240,968 (30,121)
     - 사설잡지: 56,055 (7,007)
  - AI hub 한국어대화요약: 280,000 (35,000)
     - 개인및관계: 71,408 (8,926)
     - 미용과건강: 16,520 (2,065)
     - 상거래(쇼핑): 25,480 (3,185)
     - 시사교육: 14,840 (1,855)
     - 식음료: 29,400 (3,675)
     - 여가생활: 35,840 (4,480)
     - 일과직업: 20,152 (2,519)
     - 주거와 생활: 45,640 (5,705)
     - 행사: 20,720 (2,590)
- default로 Data/Training/train_v1.csv, data/Validation/dev_v1.csv 형태로 저장        
- 데이터 열 이름을 text와 abstractive로 통일

|text|abstractive|
|:----:|:-----------:|
|문서원문|요약문|

***

## 2. Hyper-parameter tuning

- Batch size
  - trainer 생성시 `auto_scale_batch_size='power'`로 설정
  - 2, 4, 8, 16, 32..등 batch sizee 늘려가며 훈련 시도
  - 최적의 batch size 탐색 후 16으로 사용           
    <img width="600" src="https://user-images.githubusercontent.com/44887886/131794480-3320f7bd-53f1-4c1a-849f-2738926755a3.png">

- Learing rate
  - 자동으로 초기 최적의 학습률을 찾아주는 lr finder 함수 사용
  - default learing rate: 3e-5   
  - 가장 낮은 손실의 학습률이 아니라 가장 급격한 하향 기울기를 가진 위치의 학습률 선택                        
  
  <img width="800" aligh='left' src="https://user-images.githubusercontent.com/44887886/131793752-796da55e-08ce-4a09-a017-b2956c65b1ad.png">

- Num workers
  - cpu 작업에 사용할 코어의 개수
  - 해당 환경에서 사용가능한 코어 개수 확인 후 적절하게 10으로 설정
  
***

## 3. 함수 구현

- Early Stopping
  - 학습 도중 validation loss가 특정 지점 이후 계속 올라감에도 불구하고 설정한 max epoch까지 학습이 진행 -> **overfitting**        
    <img width="550" src="https://user-images.githubusercontent.com/44887886/131795045-925b7ca4-1adc-438b-962b-a7007a871c1b.png">
  - Epoch 하나 끝날 때마다 val_loss 계산
  - 연속으로 10번 이상 val_loss 증가 시 train 중단     
     
- Test_step 및 Test_epoch_end  
  - test_step: 각 배치의 평균 루지 스코어를 구한 후 반환
  - test_epoch_end: epoch 하나가 끝나기 전에, test_step 함수에서 반환된 평균 루지스코어들의 평균을 계산하여 반환
  - 모델의 최종 성능은 **rouge.csv** 파일의 형태로 저장    

***

## 4. How To Train
- 실행 환경: gpu 2개 사용 (multi-gpu)
- 모델 새로 생성
```
!python src/KoBART/train.py \
--train_file='rsc/Data/Training/AI_hub/한국어대화요약/train_v1.csv' \
--test_file='rsc/Data/Validation/AI_hub/한국어대화요약/dev_v1.csv' \
--mode='train' \
--batch_size=16 \
--num_workers=10 \
--gradient_clip_val=1.0 \
--gpus=2 \
--accelerator=ddp \
--max_epochs=50 \
--default_root_dir='rsc/By_domain_ckpt/aihub한국어대화요약'
```
- 모델 추가 학습
```
!python src/KoBART/train.py \
--train_file='rsc/Data/Training/AI_hub/한국어대화요약/train_v1.csv' \
--test_file='rsc/Data/Validation/AI_hub/한국어대화요약/dev_v1.csv' \
--checkpoint_path='rsc/By_domain_ckpt/dacon신문기사/kobart_summary-model_chp/epoch=02-val_loss=1.326.ckpt' \
--hparams_file='rsc/By_domain_ckpt/dacon신문기사/tb_logs/default/version_0/hparams.yaml' \
--mode='train' \
--batch_size=16 \
--num_workers=10 \
--gradient_clip_val=1.0 \
--gpus=2 \
--accelerator=ddp \
--max_epochs=50 \
--default_root_dir='rsc/By_domain_ckpt/aihub한국어대화요약_dacon신문기사'
```

***

## 5. Model Performance

- Test Data 기준으로 rouge score를 산출함
- Score 산출 방법은 [rouge 패키지](https://pypi.org/project/rouge/)를 사용함
- 한국어대화요약 모델로 학습한 경우 Rouge-1 f1 score : 0.12785


***

## 6. Visualization

<img width="650" src="https://user-images.githubusercontent.com/44887886/131796878-0a9ebc7a-a1e5-4361-8879-775244be4eaf.png">

***

## 7. 결과 분석

- Hyper-parameter에 따른 모델 성능 차이
  - 아래의 표는 Rouge-1의 f1 score를 나타냄
  - train: dacon 신문기사/ test: dacon 신문기사      
    |Hyper-parameter|#1|#2|#3|#4|
    |:---:|:---:|:---:|:---:|:---:|
    |Batch size|16|16|4|4|
    |Num workers|10|10|4|4|
    |Learning rate|3e-5|Lr finder|3e-5|Lr finder|
    |Time(1 epoch)|7min|9m 40s|20m|23min|
    |Validation loss|1.326|1.333|1.358|1.350|
    |Rouge-1 (f1)|0.32022|0.30888|0.32362|0.31350|
  
    - batch size와 num worker을 최대한 크게 사용했을 때 loss가 낮음
    - learing rate를 기존 값으로 이용했을 때와 lr finder을 사용했을 때, rouge score의 점수 차이는 미비
    - 따라서 이후의 결과 분석은 batch size 16, num worker 10, learing rage 3e-5(default)의 크기로 학습 진행한 결과를 나타냄
  
- 데이터의 형식이 각기 다른 경우
  - 아래의 표는 Rouge-1의 f1 score를 나타냄
    |Test set\Train set|신문기사|법률문서|신문기사+법률문서|법률문서+신문기사|
    |:---:|:---:|:---:|:---:|:---:|
    |신문기사|0.32022|0.18859|0.24843|0.33576|
    |법률문서|0.29876|0.37347|0.39297|0.38668|
    |사설잡지|0.16879|0.11018|0.16563|0.18492|
    
  - 신문기사와 같이 기자마다 혹은 신문사마다 **작성 형태가 다른 데이터(데이터 특성이 뚜렷하지 않은 데이터)**의 경우
      - train: 신문기사/ test: 신문기사 vs train: 신문기사/ test: 법률문서 -> 성능 차이 ㅇ
      - train: 신문기사+법률문서 test: 신문기사 vs train: 법률문서+신문기사 -> 성능 차이 ㅇ
  - 법률문서와 같이 **작성 형태가 일관된 데이터(데이터 특성이 뚜렷한 데이터)**의 경우
      - train: 법률문서/ test: 법률문서 vs train: 법률문서/ test: 신문기사 -> 성능 차이 미미

- 데이터의 도메인(주제)이 각기 다른 경우 (데이터 형식은 대화 형태로 모두 같음)
  - 아래의 표는 Rouge-1의 f1 score를 나타냄
  - 한국어대화요약(종합)은 모든 도메인의 데이터를 합친 후, 학습한 모델을 의미

    |Test set\Train set|상거래(쇼핑)|건강과 미용|일과 직업|한국어대화요약(종합)|
    |:---:|:---:|:---:|:---:|:---:|
    |상거래(쇼핑)|0.08452|0.07614|0.05281|0.11656|
    |건강과미용|0.08530|0.11728|0.07816|0.13320|
    |일과직업|0.07478|0.08708|0.09922|0.13633|
  
  - rouge 스코의 성능이 대체적으로 비슷함
  - 즉, 도메인(주제)이 성능에 미치는 영향은 미미함

***
## 8. 인사이트 도출
  
**⭐️Bart 모델 요약 결과에 대한 전반적인 인사이트⭐️**     


- 데이터 특성이 뚜렷하지 않은 데이터로 모델을 사용할 경우  
  - 데이터 특성이 뚜렷한 다른 형식의 데이터의 추가학습은 성능 저하를 유발한다. (+ 추가 학습하는 데이터의 크기가 작더라도 지배적인 영향을 미침)
  - 학습 순서에 따라 성능이 달라진다. 마지막에 학습된 데이터가 더 강햔 영향을 미친다.
  - 따라서 신문기사와 같은 데이터의 텍스트 요약을 진행한다면, 다른 형식의 도메인 데이터는 추가 학습 하지 않는 것이 좋다.       

- 데이터 특성이 뚜렷한 데이터로 모델을 사용할 경우
    - 데이터 형식에 상관 없이, 추가학습이 미치는 영향은 미미하다.
    - 학습 순서에 따라 성능이 크게 바뀌지 않는다.
    - 따라서 법률문서와 같은 일정한 형태의 데이터의 텍스트 요약을 진행한다면, 형식에 관계 없이 다양한 데이터를 추가 학습해도 좋다.      
 
- 도메인(주제)이 다른 데이터를 추가학습할 경우
    - 도메인(주제)이 성능에 미치는 영향은 미미하므로, 데이터 형식이 같다면 주제에 상관없이 많이 학습할 수록 좋다.


**⭐️의료 도메인 상담 데이터에 적용했을 경우⭐️**

1. 같은 형식의 도메인 학습
   - 대화체의 도메인 학습이 필요하다. 즉, 대화체가 아닌 문어체 같은 데이터를 추가학습할 경우 성능이 저하될 것으로 예상된다.
   - 비슷한 길이의 텍스트 학습이 필요하다.
2. 다양한 주제의 도메인 학습
   - 실제 적용될 상담 데이터와 비슷한 형식의 데이터라면 대화의 주제가 달라도, 즉 다른 주제의 상담 내역을 많이 학습한다면 성능향상에 도움이 될 것으로 예상된다.

***
## Reference

- [KoBART](https://github.com/SKT-AI/KoBART)
- [KoBART-summarization](https://github.com/seujung/KoBART-summarization)
      
