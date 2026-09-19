# Synapse1 Mini

Synapse1 Mini는 2021년부터 기획되었고, 2022년부터 개발을 시작한 개인 연구 프로젝트로, 언어의 인과적 관계를 학습하는 Causal Language Model(CLM)을 직접 설계하고 구현하는 것을 목표로 합니다.

오랜 기간 개인 컴퓨터에서 개발해 온 프로젝트의 연구 과정과 결과물을 정리하여 Github에 2026년부터 공개하고 있습니다.

## Overview

Synapse1 Mini는 기존 Transformer 구조를 그대로 사용하는 것에 그치지 않고, 작은 언어 모델에서도 효율적인 학습과 추론이 가능하도록 **모델 아키텍처 자체를 연구하고 설계하는 것**을 주요 목표로 합니다.

이 저장소에는 Synapse1 프로젝트에서 연구한 모델 구조와 학습 코드, 실험 결과 및 관련 자료가 단계적으로 추가될 예정입니다.

## Goals

Synapse1 Mini는 다음과 같은 목표를 가지고 개발하고 있습니다.

- 작은 규모의 언어 모델에서도 효율적으로 학습할 수 있는 아키텍처 연구
- 기존 Transformer 구조를 바탕으로 새로운 구조적 아이디어를 실험
- 제한된 컴퓨터 자원에서도 연구와 실험을 지속할 수 있는 sLM 개발
- 모델의 크기를 키우는 것뿐만 아니라, 아키텍처 자체의 개선을 통한 성능 향상 연구
- 직접 설계한 아키텍처를 구현하고 실제 학습을 통해 검증

## Philosophy

Synapse1 Mini는 단순히 더 큰 모델을 만드는 것을 목표로 하지 않습니다.

<!-- 모델의 성능은 파라미터 수와 학습 데이터의 양만으로 결정되는 것이 아니라,
모델이 정보를 처리하는 방식과 학습하는 방식 자체에도 영향을 받는다고 생각합니다. -->
모델의 성능은 파라미터 수만으로 결정되는 것이 아니라,
정보 처리 방식, 학습 데이터와 방법에 영향을 받는다고 생각합니다.

<!-- 따라서 Synapse1에서는 기존 아키텍처를 그대로 사용하는 것보다,
직접 구조를 설계하고 다양한 아이디어를 실험하며 모델의 학습 방식 자체를 연구하는 것을 중요하게 생각합니다. -->
따라서 Synapse1에서는 기존 아키텍처를 그대로 사용하는 것보다,
직접 구조를 설계하고 다양한 아이디어를 실험해보며 모델의 정보 처리 방식과 학습 방법 자체를 연구하는 것을 중요하게 생각합니다.

<!-- 또한 거대한 컴퓨팅 자원을 전제로 하지 않고,
제한된 환경에서도 새로운 아이디어를 실제 모델에 적용하고 검증할 수 있는 연구를 지향합니다. -->
또한 거대한 컴퓨팅 자원을 전제로 하지 않고,
제한된 컴퓨팅 환경에서도 새로운 아이디어와 학습 방법을 실제 모델에 적용하고 검증할 수 있는 연구를 지향합니다.
추후 개발이 완료된 모델은 실사용 가능한 학습을 거친 후 배포될 예정입니다.

## Architecture

<!-- 현재 공개된 Synapse1 Mini의 아키텍처는 다음과 같습니다. -->
Synapse1 Mini는 Llama 기반 Transformer를 사용하지만, 기존 Transformer의 구성 요소를 그대로 사용하는 것을 목표로 하지 않습니다.

각 구성 요소가 언어 모델의 학습과 정보 처리에 어떤 영향을 주는지 실험하고,
필요에 따라 새로운 구조나 기존 구조의 변형을 적용하는 방향으로 개발하고 있습니다.

현재 아키텍처에는 다음과 같은 기술을 적용하고 있습니다.

- Grouped-query Attention
- Sliding Window Attention
- SiTU-GLU
- Series and Parallel Operation Block
- Factorized Embedding
- Multi Layer LM Head
- Value Exclusive Attention
- QK-Norm, RoPE
- Pre-Norm, RMSNorm

![Synapse1 Mini Architecture Diagram](synapse1-mini.png)
<sub>본 아키텍처 다이어그램은 Sebastian Raschka의 [LLM Architecture Gallery](https://sebastianraschka.com/llm-architecture-gallery/)의 시각적 표현 방식을 참고하여 Synapse1 프로젝트를 위해 새롭게 제작되었습니다. 원본 이미지를 복제한 것이 아닙니다.</sub>
<sub>Reference: 
- [Sebastian Raschka](https://sebastianraschka.com)
- [LLM Architecture Gallery](https://sebastianraschka.com/llm-architecture-gallery)</sub>

## Intended Use

Synapse1 Mini는 연구 및 실험을 목적으로 개발하고 있습니다.

주요 사용 형태는 다음과 같습니다.

- sLM 연구 및 아키텍처 실험
- 로컬 환경에서의 모델 학습 및 추론
- 새로운 아키텍처 아이디어의 검증
- 개인 및 연구 목적의 언어 모델 실험
- 추후 공개되는 Synapse1 모델을 활용한 애플리케이션 개발

## Project Status

현재 프로젝트는 개발 및 실험이 진행 중입니다.

| 항목 | 상태 |
|---|---|
| Architecture design | 진행 중 |
| Architecture diagram | 완료 (변경사항 적용) |
| Model implementation | 진행 중 |
| Training | 진행 중 |
| Evaluation | 예정 |
| Pretrained Weights | 예정 |
| Documentation | 진행 중 |

저장소에 공개되는 내용은 개발 과정에 따라 순차적으로 추가될 예정입니다.

## License

라이선스 및 각 파일의 사용 조건은 프로젝트의 코드, 모델 가중치, 문서 및 기타 자료의 특성에 따라 별도로 명시할 예정입니다.

## Citation

Synapse1 Mini의 아키텍처 또는 연구 내용을 참고하거나 이를 기반으로 연구를 진행하는 경우, 해당 프로젝트를 인용해주세요.

#### 잡담

초등학생 때부터 기획과 동시에 코딩을 배우며, 중학생 때엔 Transformer와 PyTorch를 공부하고, 고등학생이 된 지금 모델을 구현하는 데까지 이르렀습니다.

한 문장으로 압축하기 힘들 정도로 하루 5시간 이상씩 AI와 코딩을 공부하며 새로운 기법들과 논문들, 영상을 찾아보며 지금의 모델을 만들어온 것이 매우 기쁩니다.

여태 컴퓨터에 쌓인 모델 파일들과 수많은 Pretrained 모델들, 그리고 성공과 실패를 보며 혼자만의 시간에 빠져있다가 지금에서야 Github를 사용하여 차근히 공개합니다.

아직 AI는 계속 발전하고, 개발 환경이 부족하여 완벽한 sLM을 만들지는 못했지만, 계속 발전하여 언젠가 완벽에 가까운 sLM을 만들도록 노력하겠습니다.