"""
Titanic 機器學習核心：多模型訓練 / 超參數搜尋 / 指標 / 歷史 / 預測。

跟 Flask (app.py) 分開，web 層只負責轉發。
- 一鍵訓練會在背景 thread 跑，透過 get_status() 輪詢進度
- 三種模型各自 GridSearchCV 調超參數，自動挑最佳者存檔
- 存檔：model/titanic_model.joblib（最佳模型）、meta.json（完整結果）、history.json（歷次紀錄）
"""

import os
import re
import json
import time
import sqlite3
import datetime
import threading
import warnings

# SVC(probability=True) 在 sklearn 1.9 標記 deprecated（1.11 才移除），現在仍可用，先靜音避免 console 洗版
warnings.filterwarnings("ignore", message=".*probability.*parameter was deprecated.*")

import numpy as np
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split, GridSearchCV, ParameterGrid
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, roc_curve,
)

# ------------------------------------------------------------------
# 路徑與欄位設定
# ------------------------------------------------------------------

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# 模型存放在專案根目錄的 models/（與 titanic_restful_project/ 同層）
MODEL_DIR = os.path.join(os.path.dirname(BASE_DIR), "models")
MODEL_PATH = os.path.join(MODEL_DIR, "titanic_model.joblib")
META_PATH = os.path.join(MODEL_DIR, "meta.json")
HISTORY_PATH = os.path.join(MODEL_DIR, "history.json")
DB_PATH = os.path.join(BASE_DIR, "my_db.db")

# 進模型的特徵：含 Feature Engineering 衍生出來的 FamilySize / Title / IsAlone
NUMERIC = ["Age", "Fare", "FamilySize"]
CATEGORICAL = ["Pclass", "Sex", "Embarked", "Title", "IsAlone"]   # Pclass 當類別處理（1/2/3）
FEATURES = NUMERIC + CATEGORICAL
TARGET = "Survived"

# 消融對照用的「無特徵工程」原始欄位（不含 Title / FamilySize / IsAlone）
RAW_NUMERIC = ["Age", "Fare", "SibSp", "Parch"]
RAW_CATEGORICAL = ["Pclass", "Sex", "Embarked"]
RAW_FEATURES = RAW_NUMERIC + RAW_CATEGORICAL

RANDOM_STATE = 42


# ------------------------------------------------------------------
# 資料前處理：確保「訓練」與「預測」用完全一致的型別
#   Pclass 一律轉成字串，避免 int(1) / float(1.0) / str('1') 在 OneHot 對不上。
# ------------------------------------------------------------------

# 從姓名抽出稱謂：Titanic 經典特徵，Mr/Mrs/Miss/Master 之外一律歸為 Rare
_TITLE_MAP = {
    "Mr": "Mr", "Mrs": "Mrs", "Miss": "Miss", "Master": "Master",
    "Mlle": "Miss", "Ms": "Miss", "Mme": "Mrs",
}


def _extract_title(name):
    if not isinstance(name, str):
        return np.nan
    m = re.search(r",\s*([^.]+)\.", name)     # "Braund, Mr. Owen Harris" → "Mr"
    if not m:
        return "Rare"
    return _TITLE_MAP.get(m.group(1).strip(), "Rare")


# Feature Engineering：從原始欄位衍生新特徵
def _engineer(df):
    df = df.copy()
    sib = pd.to_numeric(df["SibSp"], errors="coerce").fillna(0) if "SibSp" in df else 0
    par = pd.to_numeric(df["Parch"], errors="coerce").fillna(0) if "Parch" in df else 0
    df["FamilySize"] = sib + par + 1                       # 同行人數（含本人）
    df["IsAlone"] = np.where(df["FamilySize"] == 1, "1", "0")
    # Title：有 Name 就從 Name 抽（訓練 / 批次）；否則用外部直接給的 Title（單筆預測）
    if "Name" in df.columns and df["Name"].notnull().any():
        df["Title"] = df["Name"].apply(_extract_title)
    elif "Title" not in df.columns:
        df["Title"] = np.nan
    return df


def _prep_frame(df):
    df = _engineer(df)
    for c in NUMERIC:
        if c in df:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    if "Pclass" in df:
        df["Pclass"] = df["Pclass"].apply(
            lambda v: str(int(float(v))) if pd.notnull(v) and str(v) != "" else np.nan
        )
    for c in ["Sex", "Embarked", "Title", "IsAlone"]:
        if c in df:
            df[c] = df[c].apply(
                lambda v: str(v) if pd.notnull(v) and str(v) != "" else np.nan
            )
    for c in FEATURES:                    # 缺欄位補上，讓 batch CSV 可以少給欄位
        if c not in df:
            df[c] = np.nan
    return df


def _build_preprocessor(numeric_cols=NUMERIC, categorical_cols=CATEGORICAL):
    numeric = Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),      # LogReg / SVM 需要標準化，RF 不受影響
    ])
    categorical = Pipeline([
        ("impute", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])
    return ColumnTransformer([
        ("num", numeric, numeric_cols),
        ("cat", categorical, categorical_cols),
    ])


def _prep_raw(df):
    """只做型別正規化、不做特徵工程，給消融對照用。"""
    df = df.copy()
    for c in RAW_NUMERIC:
        if c in df:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    if "Pclass" in df:
        df["Pclass"] = df["Pclass"].apply(
            lambda v: str(int(float(v))) if pd.notnull(v) and str(v) != "" else np.nan)
    for c in ["Sex", "Embarked"]:
        if c in df:
            df[c] = df[c].apply(lambda v: str(v) if pd.notnull(v) and str(v) != "" else np.nan)
    for c in RAW_FEATURES:
        if c not in df:
            df[c] = np.nan
    return df


def _model_specs():
    """每個模型：(分類器, 超參數搜尋格) — grid 刻意精簡，891 筆秒級跑完。"""
    return {
        "LogisticRegression": (          # baseline：先用最簡單的模型建立比較基準
            LogisticRegression(max_iter=1000),
            {
                "clf__C": [0.1, 1.0, 10.0],
            },
        ),
        "DecisionTree": (
            DecisionTreeClassifier(random_state=RANDOM_STATE),
            {
                "clf__max_depth": [3, 5, 10, None],
                "clf__min_samples_split": [2, 5],
            },
        ),
        "RandomForest": (
            RandomForestClassifier(random_state=RANDOM_STATE),
            {
                "clf__n_estimators": [100, 300],
                "clf__max_depth": [None, 5, 10],
                "clf__min_samples_split": [2, 5],
            },
        ),
        "SVM": (
            SVC(probability=True, random_state=RANDOM_STATE),
            {
                "clf__C": [0.5, 1.0, 10.0],
                "clf__kernel": ["rbf", "linear"],
            },
        ),
        "XGBoost": (
            XGBClassifier(random_state=RANDOM_STATE, eval_metric="logloss", verbosity=0),
            {
                "clf__n_estimators": [100, 200],
                "clf__max_depth": [2, 3, 4],
                "clf__learning_rate": [0.05, 0.1],
            },
        ),
    }


# baseline 模型名稱（給前端標記用）
BASELINE_MODEL = "LogisticRegression"

# 互動式超參數試跑的組合數上限（避免使用者一次掃太多組把伺服器卡住）
TUNE_MAX_COMBOS = 60


def run_tuning(model_name, grid):
    """互動式試跑：用使用者給的參數格對單一模型跑 GridSearch，回報最佳組合。
    這是一次性的實驗，不會覆蓋已存檔的最佳模型。"""
    specs = _model_specs()
    if model_name not in specs:
        raise ValueError("未知的模型")
    clf, _default = specs[model_name]

    clean = {f"clf__{k}": v for k, v in (grid or {}).items() if isinstance(v, list) and len(v)}
    if not clean:
        raise ValueError("請至少提供一組參數值")
    n = len(list(ParameterGrid(clean)))
    if n > TUNE_MAX_COMBOS:
        raise ValueError(f"參數組合共 {n} 組，超過上限 {TUNE_MAX_COMBOS} 組，請減少一些值")

    raw = load_data()
    X = _prep_frame(raw)[FEATURES]
    y = raw[TARGET].astype(int)
    pipe = Pipeline([("prep", _build_preprocessor()), ("clf", clf)])
    gs = GridSearchCV(pipe, clean, cv=5, scoring="accuracy", n_jobs=-1)

    t0 = time.perf_counter()
    gs.fit(X, y)
    return {
        "model": model_name,
        "best_params": _clean_params(gs.best_params_),
        "cv_accuracy": round(float(gs.best_score_), 4),
        "n_candidates": n,
        "cv_folds": 5,
        "fit_seconds": round(time.perf_counter() - t0, 2),
    }


# ------------------------------------------------------------------
# 資料讀取
# ------------------------------------------------------------------

def load_data():
    conn = sqlite3.connect(DB_PATH)
    try:
        return pd.read_sql_query("SELECT * FROM titanic", conn)
    finally:
        conn.close()


# ------------------------------------------------------------------
# 資料概覽（EDA）— 給頁面上方的「資料洞察」區塊
# ------------------------------------------------------------------

def get_data_overview():
    df = load_data()
    total = len(df)
    survived = int(df["Survived"].sum())

    def rate_by(col, src=df):
        g = src.groupby(col)["Survived"].agg(["mean", "count"])
        return [{"key": str(k), "rate": round(float(row["mean"]), 4),
                 "count": int(row["count"])} for k, row in g.iterrows()]

    # 欄位與缺失值統計
    columns = [{"name": c, "dtype": str(df[c].dtype),
                "missing": int(df[c].isnull().sum()),
                "missing_pct": round(float(df[c].isnull().sum()) / total * 100, 1)}
               for c in df.columns]

    # 年齡直方圖
    edges = list(range(0, 81, 10))
    hist, _ = np.histogram(df["Age"].dropna(), bins=edges)
    age_hist = [{"label": f"{edges[i]}-{edges[i + 1]}", "count": int(hist[i])}
                for i in range(len(hist))]

    # 前處理實際會用到的填補值
    emb_mode = df["Embarked"].mode()
    impute = {
        "Age_median": round(float(df["Age"].median()), 1),
        "Fare_median": round(float(df["Fare"].median()), 2),
        "Embarked_mode": (emb_mode.iloc[0] if len(emb_mode) else None),
    }

    # One-Hot 之後的特徵維度
    prepped = _prep_frame(df)
    prep = _build_preprocessor().fit(prepped[FEATURES])
    n_features_out = int(len(prep.get_feature_names_out()))

    # 標準化示範用：實際 fit 一個 StandardScaler 取得 mean / std
    num_imputed = prepped[NUMERIC].apply(pd.to_numeric, errors="coerce")
    num_imputed = num_imputed.fillna(num_imputed.median())
    sc = StandardScaler().fit(num_imputed)
    scaler = {c: {"mean": round(float(sc.mean_[i]), 2), "std": round(float(sc.scale_[i]), 2)}
              for i, c in enumerate(NUMERIC)}

    # 特徵工程後的存活率（證明衍生特徵有訊號）
    eng = _engineer(df)
    eng["Survived"] = df["Survived"].values
    fam = eng.copy()
    fam["FamGroup"] = np.where(fam["FamilySize"] == 1, "1（獨自）",
                               np.where(fam["FamilySize"] <= 4, "2-4 人", "5+ 人"))

    return {
        "total": total, "survived": survived,
        "survival_rate": round(survived / total, 4),
        "n_rows": total, "n_cols": len(df.columns),
        "columns": columns,
        "by_sex": rate_by("Sex"), "by_pclass": rate_by("Pclass"), "age_hist": age_hist,
        "impute": impute, "n_features_out": n_features_out, "scaler": scaler,
        "fe": {
            "by_title": rate_by("Title", eng),
            "by_isalone": rate_by("IsAlone", eng),
            "by_family": rate_by("FamGroup", fam),
        },
    }


# ------------------------------------------------------------------
# 背景訓練狀態（module 級 + lock）
# ------------------------------------------------------------------

_status = {"status": "idle"}
_lock = threading.Lock()


def get_status():
    with _lock:
        return dict(_status)


def _set_status(**kw):
    with _lock:
        _status.clear()
        _status.update(kw)


def start_training():
    """開背景 thread 訓練；已在訓練中則回 False。"""
    with _lock:
        if _status.get("status") == "training":
            return False
        _status.clear()
        _status.update(status="training", progress="準備資料…", current=0, total=len(_model_specs()))
    threading.Thread(target=_train, daemon=True).start()
    return True


def _clean_params(params):
    return {k.replace("clf__", ""): v for k, v in params.items()}


def _feature_importance(est):
    """RF/GB 用 feature_importances_，LogReg 用 |coef|，SVM(rbf) 沒有就回空。"""
    prep = est.named_steps["prep"]
    clf = est.named_steps["clf"]
    names = list(prep.get_feature_names_out())
    if hasattr(clf, "feature_importances_"):
        vals = np.asarray(clf.feature_importances_, dtype=float)
    elif hasattr(clf, "coef_"):
        vals = np.abs(np.asarray(clf.coef_[0], dtype=float))
    else:
        return []
    s = vals.sum()
    if s > 0:
        vals = vals / s          # 正規化成相對重要度（總和為 1，方便解讀成佔比）
    pairs = sorted(
        zip(names, [round(float(v), 4) for v in vals]),
        key=lambda t: t[1], reverse=True,
    )
    return [{"feature": n.split("__", 1)[-1], "value": v} for n, v in pairs[:12]]


def _evaluate(est, X_test, y_test):
    proba = est.predict_proba(X_test)[:, 1]
    pred = est.predict(X_test)
    fpr, tpr, _ = roc_curve(y_test, proba)
    return {
        "metrics": {
            "accuracy": round(accuracy_score(y_test, pred), 4),
            "precision": round(precision_score(y_test, pred, zero_division=0), 4),
            "recall": round(recall_score(y_test, pred, zero_division=0), 4),
            "f1": round(f1_score(y_test, pred, zero_division=0), 4),
            "roc_auc": round(roc_auc_score(y_test, proba), 4),
        },
        "confusion_matrix": confusion_matrix(y_test, pred).tolist(),
        "roc_curve": [[round(a, 4), round(b, 4)] for a, b in zip(fpr.tolist(), tpr.tolist())],
    }


def _train():
    """實際訓練流程（在背景 thread 執行）。"""
    try:
        df = _prep_frame(load_data())
        X, y = df[FEATURES], df[TARGET].astype(int)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE
        )

        specs = _model_specs()
        results = []
        best_result = None
        best_estimator = None
        t_start = time.perf_counter()

        for i, (name, (clf, grid)) in enumerate(specs.items(), 1):
            _set_status(status="training", progress=f"訓練 {name} 中…",
                        current=i, total=len(specs))

            pipe = Pipeline([("prep", _build_preprocessor()), ("clf", clf)])
            gs = GridSearchCV(pipe, grid, cv=5, scoring="accuracy", n_jobs=-1)

            t0 = time.perf_counter()
            gs.fit(X_train, y_train)
            fit_seconds = round(time.perf_counter() - t0, 2)
            est = gs.best_estimator_

            r = {
                "name": name,
                "is_baseline": name == BASELINE_MODEL,
                "best_params": _clean_params(gs.best_params_),
                "cv_accuracy": round(float(gs.best_score_), 4),
                "grid": {k.replace("clf__", ""): v for k, v in grid.items()},
                "n_candidates": len(list(ParameterGrid(grid))),
                "fit_seconds": fit_seconds,
                "feature_importance": _feature_importance(est),
                **_evaluate(est, X_test, y_test),
            }
            results.append(r)

            if best_result is None or r["metrics"]["accuracy"] > best_result["metrics"]["accuracy"]:
                best_result, best_estimator = r, est

        # 消融對照：固定最佳模型，改用「無特徵工程」的原始欄位、相同切分重訓一次
        _set_status(status="training", progress="特徵工程消融對照中…",
                    current=len(specs), total=len(specs))
        raw = load_data()
        Xr = _prep_raw(raw)[RAW_FEATURES]
        yr = raw[TARGET].astype(int)
        Xr_tr, Xr_te, yr_tr, yr_te = train_test_split(
            Xr, yr, test_size=0.2, stratify=yr, random_state=RANDOM_STATE)
        clf_raw, grid_raw = _model_specs()[best_result["name"]]
        pipe_raw = Pipeline([
            ("prep", _build_preprocessor(RAW_NUMERIC, RAW_CATEGORICAL)),
            ("clf", clf_raw),
        ])
        gs_raw = GridSearchCV(pipe_raw, grid_raw, cv=5, scoring="accuracy", n_jobs=-1)
        gs_raw.fit(Xr_tr, yr_tr)
        raw_est = gs_raw.best_estimator_
        raw_acc = round(float(accuracy_score(yr_te, raw_est.predict(Xr_te))), 4)
        raw_auc = round(float(roc_auc_score(yr_te, raw_est.predict_proba(Xr_te)[:, 1])), 4)
        ablation = {
            "model": best_result["name"],
            "with_fe": best_result["metrics"]["accuracy"],
            "without_fe": raw_acc,
            "delta": round(best_result["metrics"]["accuracy"] - raw_acc, 4),
            "with_fe_auc": best_result["metrics"]["roc_auc"],
            "without_fe_auc": raw_auc,
            "n_features_fe": int(len(best_estimator.named_steps["prep"].get_feature_names_out())),
            "n_features_raw": int(len(raw_est.named_steps["prep"].get_feature_names_out())),
        }

        total_seconds = round(time.perf_counter() - t_start, 2)
        os.makedirs(MODEL_DIR, exist_ok=True)
        joblib.dump(best_estimator, MODEL_PATH)

        meta = {
            "trained_at": datetime.datetime.now().isoformat(timespec="seconds"),
            "best_model": best_result["name"],
            "best": best_result,
            "results": results,
            "n_train": int(len(X_train)),
            "n_test": int(len(X_test)),
            "n_features": len(X.columns),
            "n_features_out": int(len(best_estimator.named_steps["prep"].get_feature_names_out())),
            "cv_folds": 5,
            "search_method": "Grid Search",
            "total_seconds": total_seconds,
            "ablation": ablation,
        }
        with open(META_PATH, "w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)
        _append_history(meta)

        _set_status(status="done", progress="完成", current=len(specs),
                    total=len(specs), meta=meta)
    except Exception as e:                       # 訓練失敗要讓前端看得到
        _set_status(status="error", progress=str(e))


# ------------------------------------------------------------------
# 模型資訊 / 歷史
# ------------------------------------------------------------------

def get_model_info():
    if not os.path.exists(META_PATH):
        return None
    with open(META_PATH, encoding="utf-8") as f:
        return json.load(f)


def get_history():
    if not os.path.exists(HISTORY_PATH):
        return []
    with open(HISTORY_PATH, encoding="utf-8") as f:
        return json.load(f)


def _append_history(meta):
    hist = get_history()
    hist.insert(0, {
        "trained_at": meta["trained_at"],
        "best_model": meta["best_model"],
        "accuracy": meta["best"]["metrics"]["accuracy"],
        "roc_auc": meta["best"]["metrics"]["roc_auc"],
        "models": [{"name": r["name"], "accuracy": r["metrics"]["accuracy"]}
                   for r in meta["results"]],
    })
    with open(HISTORY_PATH, "w", encoding="utf-8") as f:
        json.dump(hist[:50], f, ensure_ascii=False, indent=2)   # 只留最近 50 筆


# ------------------------------------------------------------------
# 預測
# ------------------------------------------------------------------

_model_cache = {"mtime": None, "model": None}


def _load_model():
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError("模型尚未訓練，請先按「一鍵訓練」。")
    mtime = os.path.getmtime(MODEL_PATH)
    if _model_cache["mtime"] != mtime:                # 換模型後自動重載
        _model_cache["model"] = joblib.load(MODEL_PATH)
        _model_cache["mtime"] = mtime
    return _model_cache["model"]


def predict_one(record):
    model = _load_model()
    X = _prep_frame(pd.DataFrame([record]))[FEATURES]
    proba = float(model.predict_proba(X)[:, 1][0])
    return {"survived": int(proba >= 0.5), "probability": round(proba, 4)}


def predict_batch(df):
    model = _load_model()
    X = _prep_frame(df)[FEATURES]
    proba = model.predict_proba(X)[:, 1]
    out = df.copy()
    out.insert(0, "Survived_pred", (proba >= 0.5).astype(int))
    out.insert(1, "Survival_probability", np.round(proba, 4))
    return out


# ------------------------------------------------------------------
# 自我測試：python ml.py （需要同目錄有 my_db.db）
# ------------------------------------------------------------------

if __name__ == "__main__":
    print("訓練中…")
    _train()                                     # 同步跑一輪
    st = get_status()
    assert st["status"] == "done", st

    info = get_model_info()
    assert info and info["best_model"] in _model_specs()
    assert len(info["results"]) == 5

    one = predict_one({"Pclass": 3, "Sex": "male", "Age": 22, "SibSp": 1,
                       "Parch": 0, "Fare": 7.25, "Embarked": "S", "Title": "Mr"})
    assert 0.0 <= one["probability"] <= 1.0 and one["survived"] in (0, 1), one

    batch = predict_batch(pd.DataFrame([
        {"Pclass": 1, "Sex": "female", "Age": 38, "SibSp": 1, "Parch": 0,
         "Fare": 71.28, "Embarked": "C", "Name": "Cumings, Mrs. John Bradley"},
    ]))
    assert "Survival_probability" in batch.columns
    assert os.path.exists(HISTORY_PATH)

    print(f"OK  最佳模型={info['best_model']}  "
          f"準確率={info['best']['metrics']['accuracy']}  "
          f"單筆預測={one}")
