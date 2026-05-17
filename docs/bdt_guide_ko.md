# BDT 학습 가이드 (한국어)

H→ZZ→4μ BDT 프로젝트의 학습 워크플로우 정리.
대상 파일: `src/train.py`, `src/features.py`, `config/selection.yaml`.

---

## 1. BDT 학습하는 방법

학습 진입점은 `python -m src.train`. 프로젝트 루트에서 실행:

```bash
python -m src.train \
  --samples config/samples.yaml \
  --config  config/selection.yaml \
  --ntuples data/ntuples/ \
  --out     data/models/bdt_v1.json
```

내부 동작 순서 (`src/train.py:105`):

1. **데이터 로드** — `data/ntuples/` 안의 모든 `*_features.parquet`을 읽고
   signal은 `label=1`, background는 `label=0`을 부여 (`src/train.py:31`).
2. **가중치 계산**
   - `xsec_weight` = lumi × σ × genWeight / Σgw
     → 물리적 yield 가중치. **평가용** (ROC, yield plot 등).
   - `train_weight` → 샘플별로 동일 기여 + signal/bkg 클래스 균형 + 평균 1로 정규화.
     → **학습용** (xgboost의 `sample_weight`).
   - `train_weight`를 xsec로 그대로 쓰지 않는 이유: DY M-50은 σ가 6225 pb로 매우 커서
     4μ 통과 이벤트 하나당 가중치가 qqZZ 대비 약 10⁷배가 됨 → BDT가 DY만 학습하고
     val AUC가 0.5로 붕괴함 (`src/train.py:38` 주석 참조).
3. **train/val/test 분할** — `train_test_split`로 stratified split (`src/train.py:125`).
   비율은 `config/selection.yaml`의 `train.test_size`, `train.val_size`로 조정.
4. **xgboost 학습** — `XGBClassifier.fit(...)`, val set 기준 early stopping
   (`src/train.py:144`).
5. **저장** — 모델은 JSON으로 (`data/models/bdt_v1.json`),
   학습 곡선은 `plots/training_history.png`로 저장.

마지막에 weighted test AUC가 로그로 출력됨 (`CLAUDE.md` 참고).

---

## 2. 파라미터 수정 방법

모든 BDT 하이퍼파라미터는 **`config/selection.yaml`**의 `bdt:` 블록에 있음
(`config/selection.yaml:32`):

```yaml
bdt:
  n_estimators:    500     # 최대 boosting round (트리 개수)
  max_depth:         5     # 트리 깊이 — 클수록 표현력↑, 과적합 위험↑
  learning_rate:     0.05  # 트리당 shrinkage — 작을수록 트리 더 많이 필요
  subsample:         0.8   # 트리당 사용하는 이벤트 비율 (stochastic GBT)
  colsample_bytree:  0.8   # 트리당 사용하는 feature 비율
  min_child_weight:  5     # leaf의 최소 weight 합 — 정규화
  reg_alpha:         0.0   # leaf weight L1 정규화
  reg_lambda:        1.0   # leaf weight L2 정규화
  early_stopping_rounds: 30
  eval_metric:    "auc"

train:
  test_size:       0.25    # 최종 평가용 hold-out 비율
  val_size:        0.10    # early stopping용 hold-out 비율 (train에서 분리)
  seed:              42
  use_gpu:        true     # GPU 미존재 시 자동으로 CPU fallback
```

튜닝 절차: **YAML 숫자를 수정 → `python -m src.train …` 재실행**.
코드 수정 불필요. `train.py`가 `bdt_cfg = dict(cfg["bdt"])`로 읽어서
(`src/train.py:140`) `XGBClassifier(**bdt_cfg, …)`에 그대로 전개함.

이벤트 선택 컷 (HLT, μ pT/η/iso, Z1/Z2 mass window, m4ℓ window 등)은
같은 파일의 `bdt:` 위쪽에 있지만, 이건 **skim 단계**(Phase 1)에 영향을 줌.
즉, 이 값들을 바꾸면 `python -m src.skim …`을 다시 돌려야 적용됨.

### 자주 만지게 되는 파라미터 직관

| 파라미터 | 효과 |
|---|---|
| `max_depth` ↑ | 모델 capacity ↑, 과적합 위험 ↑ |
| `learning_rate` ↓ + `n_estimators` ↑ | 더 안정적, 학습 시간 ↑ |
| `subsample`, `colsample_bytree` ↓ | 정규화 효과 ↑ |
| `min_child_weight` ↑ | 작은 leaf 억제 (정규화) |
| `reg_lambda` ↑ | leaf 값 부드럽게 |

---

## 3. 물리량 (pT 등) 추출 방법

### 3.1 용어

- **pT** = transverse momentum = 횡운동량.
  빔축에 수직인 운동량 성분: `pT = sqrt(px² + py²)`.
- 하드론 충돌기에서는 partons의 longitudinal momentum (pz)을 알 수 없으므로
  **transverse plane에서 균형이 맞는 pT가 표준 변수**가 됨.
- 함께 쓰이는 변수:
  - **η (pseudorapidity)**: `η = -ln tan(θ/2)`, polar angle을 빔축 기준으로 표현
  - **φ**: azimuthal angle (transverse plane 내 각도)
  - **m**: invariant mass

### 3.2 NanoAOD에서 추출 (Phase 1 — skim)

`src/skim.py`가 `uproot`로 NanoAOD ROOT 파일을 읽음. 사용하는 branch:

- `Muon_pt`, `Muon_eta`, `Muon_phi`, `Muon_mass`
- `Muon_charge`, `Muon_looseId`, `Muon_pfRelIso04_all`, `Muon_dxy`, `Muon_dz`, `Muon_sip3d`
- `HLT_*` (트리거)
- `genWeight` (MC 가중치)

선택 기준 통과한 4μ를 pT 내림차순으로 정렬해서 다음 컬럼으로 parquet에 저장:
- `pt_mu1`, `pt_mu2`, `pt_mu3`, `pt_mu4`
- `eta_mu1`…`eta_mu4`, `phi_mu1`…`phi_mu4`, `mass_mu1`…`mass_mu4`, `charge_mu1`…`charge_mu4`
- Z pairing 결과: `Z1_idx1`, `Z1_idx2`, `Z2_idx1`, `Z2_idx2`, `mZ1`, `mZ2`, `m4l`
- `genWeight` (MC) 또는 `1.0` (data)

### 3.3 합성 물리량 계산 (Phase 2 — features)

`src/features.py::add_basic_kinematics` (`src/features.py:130`)에서
4-vector 라이브러리(`vector`)로 처리:

```python
mu = _muon_p4s(events)              # pt/eta/phi/m → 4-vector 배열 4개
Z1 = mu_z1a + mu_z1b                # 4-vector 합 = Z1 4-momentum
Z2 = mu_z2a + mu_z2b
H4 = mu[0] + mu[1] + mu[2] + mu[3]  # 4μ 시스템 ≈ Higgs candidate

new = {
    "pt4l":  H4.pt,                 # 4μ 시스템의 pT
    "pt_Z1": Z1.pt,                 # Z1의 pT
    "pt_Z2": Z2.pt,
    "pt_Z1_over_m4l": Z1.pt / m4l,  # 무차원 비율
    "pt_Z2_over_m4l": Z2.pt / m4l,
    "dR_Z1Z2": Z1.deltaR(Z2),
    ...
}
```

`vector` 라이브러리(`src/features.py:24`의 `vector.register_awkward()`)가
4-벡터 객체에 `.pt`, `.eta`, `.phi`, `.mass`, `.deltaR(...)`, `.boostCM_of_p4(...)` 등을
attribute로 노출시켜 줌 → 직접 `sqrt(px²+py²)` 쓸 필요 없음.

### 3.4 Helicity angles (`src/features.py:163`)

HZZ에서 spin-0 신호를 비공명 ZZ background와 구분하는 각도 변수 5개:
`cos_theta_star`, `cos_theta1`, `cos_theta2`, `Phi`, `Phi1`.
계산 방식은 muon 4-vector를 H, Z1, Z2 rest frame으로 boost한 뒤 단위벡터들의
내적/cross product로 구함. 정의는 arXiv:1208.4018 참고.

### 3.5 Feature → BDT 입력 흐름

`src/features.py:33`의 `FEATURES` 리스트가 BDT 입력의 정식 정의.
`src/train.py:119`에서 정확히 이 컬럼만 골라 X로 사용:

```python
X = df[FEATURES].to_numpy(dtype=np.float32)
```

### 3.6 새 물리량을 BDT 입력에 추가하려면

1. `src/features.py`의 `add_basic_kinematics` (또는 새 함수)에서 컬럼 계산.
2. 같은 파일 `FEATURES` 리스트에 컬럼 이름 추가.
3. Phase 2 재실행: `python -m src.features --in data/ntuples/ --out data/ntuples/`
4. Phase 3 재실행: `python -m src.train …`

---

## 4. 물리 개념 ↔ 컬럼 이름 빠른 매핑

| 물리 개념 | 컬럼 |
|---|---|
| muon 횡운동량 (pT) | `pt_mu1`…`pt_mu4` (pT 내림차순) |
| muon pseudorapidity | `eta_mu1`…`eta_mu4` |
| Z 후보 invariant mass | `mZ1`, `mZ2` |
| 4μ (Higgs 후보) invariant mass | `m4l` |
| 4μ 시스템 pT, η | `pt4l`, `eta4l` |
| Z1, Z2 pT | `pt_Z1`, `pt_Z2` |
| 두 Z의 ΔR | `dR_Z1Z2` |
| pT 비율 | `pt_Z1_over_m4l`, `pt_Z2_over_m4l` |
| HZZ 5-angle 변수 | `cos_theta_star`, `cos_theta1`, `cos_theta2`, `Phi`, `Phi1` |

---

## 5. 학습 결과물 위치

- 모델: `data/models/bdt_v1.json` (xgboost native JSON 포맷)
- 학습 곡선: `plots/training_history.png`
- Phase 4 (평가) 결과: `plots/`의 ROC/feature importance/overtraining 플롯 +
  `reports/phase4_summary.md`
