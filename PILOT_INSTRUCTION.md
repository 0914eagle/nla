# NLA × Transcoder Grounding 파일럿 — Coding Agent 지침서

연구 목표: NLA가 Gemma-3-12B의 layer 32 residual stream에서 생성한
자연어 설명을, 같은 위치의 transcoder/circuit-tracer attribution으로
역추적할 수 있는지 확인한다. (mechanistic grounding 가능성 검증)

환경: GPU 서버 (A100 80GB 권장 — 12B 모델 + transcoder 동시 로드)

---

## 핵심 정보 (반드시 이 값을 사용)

```
Target model:        google/gemma-3-12b-it
NLA extraction layer: 32  (전체 48 레이어 중)
d_model:             3840

NLA checkpoints (HuggingFace):
  AV (verbalizer):    kitft/nla-gemma3-12b-L32-av
  AR (reconstructor): kitft/nla-gemma3-12b-L32-ar

Transcoder:
  circuit-tracer 라이브러리 사용
  Gemma-3 PLT (per-layer transcoder), GemmaScope-2 기반
  → layer 32 transcoder를 매칭해서 사용

라이브러리:
  NLA inference: github.com/kitft/nla-inference  (NLAClient, NLACritic)
  Transcoder:    github.com/decoderesearch/circuit-tracer
```

**중요**: NLA가 layer 32를 보므로, attribution graph의 시작점(target)도
반드시 layer 32 residual stream이어야 한다. 이게 두 도구를 잇는 핵심.

---

## 파일 1: setup.py — 환경 및 모델 로드

### 할 일

1. 패키지 설치
   ```
   pip install torch transformers accelerate
   pip install git+https://github.com/kitft/nla-inference.git
   pip install git+https://github.com/decoderesearch/circuit-tracer.git
   pip install sglang  # NLA AV inference에 권장됨 (README 참고)
   ```

2. Gemma-3-12B-IT 로드
   - `google/gemma-3-12b-it`, bfloat16, device_map="auto"
   - HF 토큰 필요 (Gemma는 gated model) → 환경변수 HF_TOKEN 확인

3. NLA 체크포인트 로드
   - nla-inference 패키지의 NLAClient 사용
   - AV: kitft/nla-gemma3-12b-L32-av
   - AR: kitft/nla-gemma3-12b-L32-ar
   - nla_meta.yaml에서 정확한 hook 위치 확인
     (resid_post인지 resid_mid인지 — grounding 정렬에 필수)

4. circuit-tracer ReplacementModel 로드
   - Gemma-3-12B용 PLT transcoder set 로드
   - `ReplacementModel.from_pretrained(model, transcoders=...)`
   - layer 32 transcoder가 포함되는지 확인

5. **정렬 검증 (가장 중요)**
   ```
   동일한 입력 텍스트에 대해:
   a. Gemma-3-12B를 직접 돌려서 layer 32 resid_post activation 추출
   b. NLA가 hook하는 위치의 activation 추출
   c. circuit-tracer가 layer 32에서 보는 activation 추출
   → 세 개가 같은 벡터인지 확인 (allclose)
   → 다르면 hook 위치(resid_post vs resid_mid) 불일치
      → 여기서 멈추고 사람에게 보고
   ```

### 출력
- 세 activation이 일치하는지 여부 (정렬 성공/실패)
- 실패 시 어디서 다른지 보고

---

## 파일 2: run_nla.py — NLA 설명 생성

### 할 일

1. 입력 텍스트 준비
   - 파일럿용 multi-hop 예시 (아래 데이터 섹션 참고)

2. 각 입력에 대해 layer 32 activation 추출
   - 특정 토큰 위치들의 residual stream
   - 보통 마지막 토큰 또는 관심 토큰

3. NLAClient로 설명 생성
   ```python
   from nla_inference import NLAClient
   client = NLAClient(av="kitft/nla-gemma3-12b-L32-av")
   explanation = client.verbalize(activation_vector)
   ```

4. NLACritic으로 reconstruction 품질 측정
   ```python
   from nla_inference import NLACritic
   critic = NLACritic(ar="kitft/nla-gemma3-12b-L32-ar")
   reconstructed = critic.reconstruct(explanation)
   mse = round_trip_mse(reconstructed, original)  # 2(1-cos)
   ```
   - MSE 낮으면 설명이 정보를 잘 담음
   - 논문 baseline: FVE 0.6~0.8

5. 저장
   - 각 (입력, 토큰 위치, activation, NLA 설명, reconstruction MSE)

### 출력
- 입력별 NLA 설명 + reconstruction MSE
- confabulation 의심 케이스 플래그
  (MSE는 낮은데 설명이 입력과 무관해 보이는 경우)

---

## 파일 3: run_attribution.py — Transcoder 역추적

### 할 일

1. setup에서 로드한 circuit-tracer ReplacementModel 사용

2. NLA가 설명한 그 토큰, 그 layer(32)의 activation을
   attribution graph의 target으로 설정
   ```python
   from circuit_tracer import attribute
   graph = attribute(
       model=replacement_model,
       prompt=input_text,
       target_layer=32,
       target_token=token_pos,
   )
   ```

3. attribution graph에서 상위 기여 feature 추출
   - layer 32의 target activation에 가장 크게 기여한
     transcoder feature들 (이전 레이어 포함)
   - 각 feature의 top-activating token (어느 입력 토큰에서 왔나)

4. 저장
   - target (입력, 토큰, layer 32)
   - top-K 기여 feature + 각 feature의 출처 토큰 + 기여도

### 출력
- NLA가 설명한 activation에 대한 attribution graph
- "이 activation은 입력의 어느 토큰들에서 왔다"는 추적 결과

---

## 파일 4: compare.py — NLA 설명 vs Attribution 대조 (파일럿 핵심)

### 할 일

이게 파일럿의 결론을 내는 부분.

1. NLA 설명(run_nla 출력)과 attribution(run_attribution 출력)을 정렬

2. 수동 대조용 리포트 생성
   ```
   각 케이스에 대해:
   - 입력 텍스트
   - NLA 설명: "이 activation은 X를 다룬다"
   - Attribution 출처: 입력의 토큰 [t1, t2, t3]가 기여
   - 그 토큰들의 의미와 NLA 설명 X가 일치하는가? (사람이 판단)
   ```

3. 간단한 자동 매칭 (보조)
   - NLA 설명의 핵심 명사 추출
   - attribution 출처 토큰과 string/semantic overlap 측정
   - 완벽할 필요 없음, 경향만 봄

### 출력 (파일럿 결론)
```
케이스별:
  NLA 설명의 claim ↔ attribution 출처가 일치하는가?

종합:
  → 일치 多: grounding 가능. 본 연구 진행
  → 불일치 多: 방법 재검토
  → 애매: 더 많은 케이스 필요
```

---

## 파일럿용 데이터

별도 데이터셋 다운로드 불필요. 직접 구성한 소수의 통제된 예시로 시작.

### 데이터 A: Multi-hop 추론 (희재님 예시 스타일)
```
"The capital of the country whose flag is blue-white-red
 vertical stripes is ___"
→ France → Paris

"The most famous landmark in the capital of the country
 with the most museums in Europe is ___"
→ France → Paris → Eiffel Tower
```
이런 multi-hop 입력 10개.
→ 중간 추론 토큰에서 NLA가 "France" "Paris" 같은
  intermediate concept을 설명하는지 보고
→ attribution이 실제로 그 개념 토큰에서 오는지 확인

### 데이터 B: Neuronpedia 데모 재현
```
Neuronpedia gemma-3 NLA 데모의 시나리오들:
  - Multi-Hop Animal
  - Secret Word
  - Inferring Country
→ 이미 NLA 설명이 공개돼 있어서 sanity check 가능
```

### 데이터 C: Confabulation 유발 (대조군)
```
activation에 없는 정보를 NLA가 지어내는지 보려고:
  매우 짧거나 모호한 입력
  → NLA가 구체적 claim을 만들면 confabulation 후보
  → attribution에 출처가 없어야 함 (가설)
```

파일럿은 데이터 A 10개 + 데이터 B 3개로 시작.
B는 NLA 설명이 이미 알려져 있어 정렬 검증에 유용.

---

## 실행 순서 및 체크포인트

```
1. setup.py
   체크: 세 activation 정렬 성공?
     실패 → hook 위치 불일치, 사람에게 보고 후 중단
     성공 → 다음

2. run_nla.py (데이터 B 먼저)
   체크: NLA 설명이 Neuronpedia 데모와 유사한가?
     아니오 → 모델/체크포인트 로드 문제
     예 → 데이터 A로 확장

3. run_attribution.py
   체크: layer 32 target으로 attribution graph가 나오는가?
     실패 → circuit-tracer target 설정 문제
     성공 → 다음

4. compare.py
   체크: NLA claim ↔ attribution 출처 일치하는가?
     → 이게 파일럿의 최종 답
```

---

## 주의사항

1. **Layer 32 고정**: 모든 단계에서 layer 32만 본다.
   NLA가 layer 32를 보므로 transcoder/attribution도 layer 32.

2. **Hook 위치 정확성**: nla_meta.yaml에서 resid_post/resid_mid 확인.
   이게 틀리면 NLA가 본 것과 다른 걸 분석하게 됨.

3. **메모리**: 12B 모델 + transcoder + AV + AR를 다 올리면 큼.
   - AV inference는 SGLang으로 분리 (README 권장)
   - 또는 단계별로 로드/언로드

4. **circuit-tracer가 Gemma-3 layer 32를 지원하는지**:
   PLT가 270M~27B 공개됐지만 12B의 모든 레이어가
   다 있는지 setup에서 먼저 확인. 없으면 가장 가까운
   레이어로 조정하거나 사람에게 보고.

5. **파일럿은 자동화보다 수동 검증 우선**:
   compare.py는 완전 자동 매칭보다 사람이 읽을 수 있는
   리포트 생성에 집중. 케이스 10개를 눈으로 보는 게 목표.
