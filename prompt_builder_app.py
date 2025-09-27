import streamlit as st
import pandas as pd
from io import StringIO

# --------------------------------------------------------------------------
# プロンプト生成ロジック部
# --------------------------------------------------------------------------
def generate_prompt(
    problem_type, ts_task_type,
    source_type, analysis_goal, data_context,
    train_path, test_path, submit_path, single_path,
    target_col, id_col, time_col,
    models, use_ensemble, tune_hyperparams,
    ts_features, include_corr, include_scaling, graphs,
    save_dir_path,
    use_feature_selection, n_top_features
    ):
    """ユーザーの選択に基づいてAIへのプロンプトを生成する関数"""

    # 問題タイプの文字列を整形
    if problem_type == "時系列":
        problem_type_full = f"時系列{ts_task_type}"
    else:
        problem_type_full = problem_type

    prompt = [
        f"### 依頼内容：機械学習を用いた「{problem_type_full}」の「{analysis_goal}」",
        "これからGoogle Colab環境で実行する、データ分析のPythonコードをステップ・バイ・ステップで生成してください。",
        "### 指示の前提条件:",
        "- **再現性の確保**: 分析の再現性が取れるよう、モデルの `random_state` は `42` に固定してください。",
        "- **可読性の向上**: 適切なコメントを追加し、可能であれば処理を関数にまとめてコードの可読性を高めてください。",
        "\n# ==================================",
        "# Step 1: 環境設定とデータ読み込み",
        "# ==================================",
        "### ライブラリのインストール",
        "!pip install japanize-matplotlib lightgbm shap holidays scikit-learn tsfresh -q",
        "\n### 主要ライブラリのインポート",
        "# (pandas, numpy, lightgbm, sklearn, matplotlib, seaborn, shap, holidays, os などをインポートしてください)",
        "\n### Google Driveのマウント",
        "from google.colab import drive", "drive.mount('/content/drive')",
        "\n### ファイルパスと保存先の設定",
    ]

    if source_type == "Kaggle形式":
        prompt.append(f"train_data_path = '{train_path}'")
        prompt.append(f"test_data_path = '{test_path}'")
        prompt.append(f"submit_template_path = '{submit_path}'")
    else:
        prompt.append(f"file_path = '{single_path}'")
    prompt.append(f"save_folder_path = '{save_dir_path}'")

    prompt.append("\n### データの読み込み")
    if source_type == "Kaggle形式":
        prompt.append("df_train = pd.read_csv(train_data_path)")
        prompt.append("df_test = pd.read_csv(test_data_path)")
    else:
        prompt.append("df = pd.read_csv(file_path)")

    prompt.extend([
        "\n# ==================================", "# Step 2: データ概要の把握", "# ==================================",
        "### データのカラム名とサンプル",
        "以下に、分析対象となるデータのカラム構成と先頭数行のサンプルを示します。これを元に分析を進めてください。",
        "--- データ概要 ---", data_context, "--- ここまで ---",
        f"\n今回の分析の目的変数は `{target_col}` です。"
    ])

    prompt.extend([
        "\n# ==================================",
        f"# Step 3: {analysis_goal}のための特徴量エンジニアリングと前処理",
        "# ==================================",
        "### 実行してほしいタスク:",
    ])

    tasks = [f"- **保存先ディレクトリの作成**: `os.makedirs(save_folder_path, exist_ok=True)` を実行してください。"]
    
    if problem_type == "時系列" and ts_task_type == "分類":
        tasks.append(f"- **データ構造の前提**: このタスクは時系列分類です。データは各サンプル（ID）ごとに時系列データを持つロングフォーマット（列: `{id_col}`, `{time_col}`, 値, `{target_col}`）であることを前提としてください。")
        tasks.append(f"- **特徴量エンジニアリング**: 各サンプルIDの時系列データから、**系列全体を要約する特徴量**を抽出してください。具体的には、平均値、標準偏差、最大値、最小値、歪度、尖度などです。可能であれば`tsfresh`ライブラリの活用も検討してください。")
    elif problem_type == "時系列" and ts_task_type == "予測":
        if source_type == "Kaggle形式": tasks.append("- **データ結合**: `df_train`と`df_test`を一度結合し、共通の前処理を実装してください。")
        tasks.append(f"- **時系列データの前処理**: `{time_col}`列をdatetime型に変換してください。")
        if ts_features: tasks.append(f"- **時系列特徴量の作成**: 以下の特徴量を作成してください。\n  - " + "\n  - ".join(ts_features))
    else:
        if source_type == "Kaggle形式": tasks.append("- **データ結合**: `df_train`と`df_test`を一度結合し、共通の前処理を実装してください。")
        tasks.append("- **カテゴリカル変数の処理**: **各カテゴリカル変数のユニーク数を調査**し、その数に応じてエンコーディングを適切に使い分けてください。特にLightGBMを使う場合は、ワンホットエンコーディングではなく、`category`型への変換とモデルへの直接指定が有効です。")

    tasks.append("- **欠損値処理**: 欠損値の有無を確認し、もし存在すれば適切な方法で処理してください。")
    if include_scaling:
        tasks.append("- **特徴量スケーリング**: `StandardScaler`などを用いて、数値特徴量のスケールを揃える処理を実装してください。")
    if include_corr:
        tasks.append("- **相関分析**: 前処理後の特徴量間の相関行列を計算し、ヒートマップで可視化してください。")

    prompt.append("\n".join(tasks))

    prompt.extend([
        "\n# ==================================",
        f"# Step 4: {analysis_goal}のためのモデル構築と評価 (初回)",
        "# ==================================",
        "### 実行してほしいタスク:",
    ])
    
    tasks = []
    tasks.append(f"- **モデル学習**: `{', '.join(models)}` を使って、予測精度が最大になるようにモデルを学習させてください。")
    if 'ロジスティック回帰/線形回帰' in models:
        tasks.append("- **予測式の表示**: 線形回帰またはロジスティック回帰モデルが学習された場合、そのモデルの係数（`coef_`）と切片（`intercept_`）を表示し、予測式を人間が理解できる形で示してください。")

    if tune_hyperparams and models and not any(m in ["ARIMA", "Prophet"] for m in models):
        cv_method = "TimeSeriesSplitを使ったクロスバリデーション" if problem_type == "時系列" and ts_task_type == "予測" else "通常のクロスバリデーション(cv=5)"
        tasks.append(f"- **ハイパーパラメータチューニング**: GridSearchCVを使い、`{models[0]}`のモデルの予測精度をさらに向上させてください。({cv_method})")
    if use_ensemble and len(models) > 1:
        tasks.append("- **アンサンブル**: 学習させた複数のモデルの予測値を平均し、アンサンブル予測を行ってください。")

    evaluation_items = []
    if problem_type == "分類" or (problem_type == "時系列" and ts_task_type == "分類"):
        evaluation_items = ["混同行列", "正解率 (Accuracy), 適合率 (Precision), 再現率 (Recall), F1スコア", "AUCスコア"]
    elif problem_type == "回帰" or (problem_type == "時系列" and ts_task_type == "予測"):
        evaluation_items = ["RMSE", "MAE", "R2スコア (決定係数)"]
        if problem_type == "時系列": evaluation_items.append("MAPE")
        
    if evaluation_items:
        tasks.append(f"\n- **モデル評価**: テストデータ（または検証データ）に対して、以下の指標を計算し、結果を報告してください。\n  - " + "\n  - ".join(evaluation_items))
        tasks.append("- **評価指標の簡単な解説**: 計算された各評価指標について、その値が一般的に「高い」のか「低い」のか、それが何を意味するのかを初心者にも分かるように簡単に解説してください。")

    tasks.append(f"\n- **可視化**: 以下のグラフを生成し、`save_folder_path`に保存してください。\n  - " + "\n  - ".join(graphs))
    if any("SHAP" in g for g in graphs):
        tasks.append("- **重要度の高い・低い特徴量のリストアップ**: SHAPの分析結果に基づき、最も重要度が高かった特徴量トップ5と、最も重要度が低かった特徴量ワースト5をリストアップしてください。")
    
    prompt.append("\n".join(tasks))

    def create_submission_tasks(is_kaggle, problem, id_col_name):
        submission_tasks = []
        if is_kaggle:
            submission_tasks.append("- **予測結果の表示と要約**: 最終的な予測結果の先頭5行と、その基本統計量（平均、標準偏差など）を表示してください。")
            submission_tasks.append(f"- **提出ファイルの作成**: `sample_submission.csv`の形式に合わせて、テストデータの予測結果を`os.path.join(save_folder_path, 'submission.csv')`として出力してください。ID列は`{id_col_name}`です。")
        elif problem == "時系列予測":
            submission_tasks.append("\n- **未来予測**: 学習したモデルを使い、未来の予測値を算出し、実績値と合わせてグラフにプロットしてください。")
        return submission_tasks

    if use_feature_selection and 'LightGBM' in models:
        prompt.extend([
            "\n# ==================================",
            "# Step 5: モデルベースの特徴量選択と再構築",
            "# ==================================", "### 実行してほしいタスク:",
            f"- **特徴量選択**: Step 4で学習したLightGBMモデルの `feature_importances_` を利用して、重要度の高い上位 **{n_top_features}個** の特徴量を選択してください。",
            "- **モデル再構築**: 選択した特徴量のみを使用して、再度LightGBMモデルを学習させてください。ハイパーパラメータはStep 4でチューニングした最適値を使用してください。",
            "- **再評価**: 新しいモデルを同じ検証データで評価し、Step 4の評価指標と比較してスコアが改善したか報告してください。",
        ])
        prompt.extend(create_submission_tasks(source_type == "Kaggle形式", problem_type_full, id_col))
    else:
        submission_tasks = create_submission_tasks(source_type == "Kaggle形式", problem_type_full, id_col)
        if submission_tasks:
            prompt.append("\n# ==================================")
            prompt.append("# Step 5: 最終予測とファイルの提出")
            prompt.append("# ==================================")
            prompt.extend(submission_tasks)
    
    prompt.append("\n---\n以上の内容で、Pythonコードを生成してください。")
    return "\n".join(prompt)

# --------------------------------------------------------------------------
# Streamlit UI部
# --------------------------------------------------------------------------
st.set_page_config(page_title="🤖 AIプロンプトビルダー", layout="wide")
st.title("🤖 AIプロンプトビルダー for Data Analysis")
st.write("データ分析のタスクをAIに依頼するための、完璧なプロンプトを自動生成します。")

if 'data_context' not in st.session_state:
    st.session_state.data_context = ""

# --- サイドバー ---
with st.sidebar:
    st.header("⚙️ 設定項目")

    with st.expander("1. 分析の目的", expanded=True):
        analysis_goal = st.radio("主な目的は？", ["予測モデルの構築", "要因分析"], horizontal=True, key="analysis_goal")
        problem_type = st.radio("問題の種類は？", ["回帰", "分類", "時系列"], horizontal=True, key="problem_type")
        
        ts_task_type = ""
        if problem_type == "時系列":
            ts_task_type = st.radio("時系列タスクを選択", ["予測", "分類"], horizontal=True, key="ts_task_type")

    with st.expander("2. データソース", expanded=True):
        source_type = st.radio("データの形式は？", ["単一ファイル", "Kaggle形式"], horizontal=True, key="source_type")
        st.caption("Google Drive内のファイルパスを入力してください。")
        if source_type == "Kaggle形式":
            train_path = st.text_input("学習データ (train.csv)", "/content/drive/MyDrive/data/train.csv")
            test_path = st.text_input("テストデータ (test.csv)", "/content/drive/MyDrive/data/test.csv")
            submit_path = st.text_input("提出用サンプル", "/content/drive/MyDrive/data/sample_submission.csv")
            single_path = ""
        else:
            single_path = st.text_input("CSVファイルのパス", "/content/drive/MyDrive/data/my_data.csv")
            train_path, test_path, submit_path = "", "", ""

    with st.expander("3. データの詳細", expanded=True):
        target_col_default = "y" if problem_type == "時系列" and ts_task_type == "予測" else "target"
        target_col = st.text_input("目的変数の列名", target_col_default, key="target_col")
        id_col = st.text_input("ID/識別子の列名", "id", key="id_col")
        
        time_col, ts_features = "", []
        if problem_type == "時系列":
            time_col_default = "datetime" if ts_task_type == "予測" else "timestamp"
            time_col = st.text_input("時系列カラムの列名", time_col_default, key="time_col")
            if ts_task_type == "予測":
                ts_features = st.multiselect(
                    "作成したい時系列特徴量",
                    ["時間ベースの特徴量", "ラグ特徴量", "移動平均特徴量", "祝日特徴量"],
                    default=["時間ベースの特徴量", "ラグ特徴量", "移動平均特徴量"],
                    key="ts_features"
                )

    with st.expander("4. データ概要", expanded=False):
        st.info("分析対象のCSVをアップロードすると、概要が自動入力されます。")
        uploaded_file = st.file_uploader("CSVをアップロードして概要を自動生成", type=['csv'], key="uploader")
        if uploaded_file:
            try:
                df_context = pd.read_csv(uploaded_file)
                buffer = StringIO()
                df_context.info(buf=buffer)
                info_str = buffer.getvalue()
                head_str = df_context.head().to_markdown()
                st.session_state.data_context = f"【df.info()】\n```\n{info_str}```\n\n【df.head()】\n{head_str}"
            except Exception as e: st.error(f"ファイル読み込みエラー: {e}")
        data_context = st.text_area("データ概要（自動入力）", st.session_state.data_context, height=300, key="data_context")

    with st.expander("5. モデルと分析手法", expanded=True):
        default_models = ["LightGBM"] if problem_type == "時系列" else ["LightGBM", "ロジスティック回帰/線形回帰"]
        models = st.multiselect("使用したいモデル", ["LightGBM", "ロジスティック回帰/線形回帰", "ランダムフォレスト", "XGBoost", "ARIMA", "Prophet"], default=default_models, key="models")

        tune_hyperparams, use_ensemble, use_feature_selection = False, False, False
        n_top_features = 50
        if analysis_goal == "予測モデルの構築":
            tune_hyperparams = st.checkbox("ハイパーパラメータチューニングを行う", True, key="tuning")
            if len(models) > 1 and not (problem_type == "時系列" and ts_task_type == "分類"):
                use_ensemble = st.checkbox("アンサンブル学習を行う", True, key="ensemble")
            if 'LightGBM' in models:
                use_feature_selection = st.checkbox("モデルベースの特徴量選択を行う", True, key="feature_selection")
                if use_feature_selection:
                    n_top_features = st.number_input("選択する上位特徴量の数", 10, 200, 50, 10, key="n_top_features")

        include_scaling = st.checkbox("特徴量スケーリングを行う", True, key="scaling")
        include_corr = st.checkbox("相関ヒートマップを作成", True, key="corr")

    with st.expander("6. 可視化と保存先", expanded=True):
        graph_options = []
        if problem_type == "分類" or (problem_type == "時系列" and ts_task_type == "分類"):
            graph_options = ["目的変数の分布", "混同行列"]
        elif problem_type == "回帰":
            graph_options = ["目的変数の分布", "実績値 vs 予測値プロット"]
        elif problem_type == "時系列" and ts_task_type == "予測":
            graph_options = ["時系列グラフのプロット", "時系列分解図", "ACF/PACFプロット"]
        
        if not any(m in ["ARIMA", "Prophet"] for m in models):
             graph_options.extend(["SHAP 重要度プロット (Bar)", "SHAP Beeswarmプロット"])
        
        graphs = st.multiselect("作成したいグラフの種類", graph_options, default=graph_options, key="graphs")
        save_dir_path = st.text_input("保存先フォルダ", "/content/drive/MyDrive/results/", key="save_path")

# --- メイン画面 ---
st.header("✅ 生成されたプロンプト")
st.info("以下のテキストをコピーして、AIアシスタントに貼り付けてください。")

if all([target_col, data_context, models]):
    generated_prompt_text = generate_prompt(
        problem_type, ts_task_type,
        source_type, analysis_goal, data_context,
        train_path, test_path, submit_path, single_path,
        target_col, id_col, time_col,
        models, use_ensemble, tune_hyperparams,
        ts_features, include_corr, include_scaling, graphs,
        save_dir_path,
        use_feature_selection, n_top_features
    )
    st.text_area("", generated_prompt_text, height=600, label_visibility="collapsed")
else:
    st.warning("サイドバーで必須項目（特に「目的変数」、「データ概要」、「使用したいモデル」）を入力・選択してください。")
