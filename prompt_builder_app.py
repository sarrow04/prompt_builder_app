import streamlit as st
import pandas as pd
from io import StringIO

# --------------------------------------------------------------------------
# プロンプト生成ロジック部 (✨✨ この部分を改善 ✨✨)
# --------------------------------------------------------------------------
def generate_prompt(
    problem_type, ts_task_type, # ts_task_type を追加
    source_type, analysis_goal, data_context,
    train_path, test_path, submit_path, single_path,
    target_col, id_col, time_col,
    models, use_ensemble, tune_hyperparams,
    ts_features, include_corr, include_scaling, graphs,
    save_dir_path,
    use_feature_selection, n_top_features
    ):
    """ユーザーの選択に基づいてAIへのプロンプトを生成する関数"""
    
    # 時系列タスクの場合は、問題タイプをより具体的にする
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
        "!pip install japanize-matplotlib lightgbm shap holidays scikit-learn tsfresh -q", # tsfreshを追加
    ]
    
    # ... (以降のStep 1, Step 2 のロジックは変更なし) ...
    prompt.extend([
        "\n### 主要ライブラリのインポート",
        "# (pandas, numpy, lightgbm, sklearn, matplotlib, seaborn, shap, holidays, os などをインポートしてください)",
        "\n### Google Driveのマウント",
        "from google.colab import drive", "drive.mount('/content/drive')",
        "\n### ファイルパスと保存先の設定",
    ])
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
    
    ### --- 改善点: 時系列分類専用の指示を追加 --- ###
    if problem_type == "時系列" and ts_task_type == "分類":
        tasks.append(f"- **データ構造の前提**: このタスクは時系列分類です。データは各サンプル（ID）ごとに時系列データを持つロングフォーマット（列: `{id_col}`, `{time_col}`, 値, `{target_col}`）であることを前提としてください。")
        tasks.append(f"- **特徴量エンジニアリング**: 各サンプルIDの時系列データから、**系列全体を要約する特徴量**を抽出してください。具体的には、平均値、標準偏差、最大値、最小値、歪度、尖度などです。可能であれば`tsfresh`ライブラリの活用も検討してください。")
    elif problem_type == "時系列" and ts_task_type == "予測":
        # (従来の時系列予測の指示)
        if source_type == "Kaggle形式": tasks.append("- **データ結合**: `df_train`と`df_test`を一度結合し、共通の前処理を実装してください。")
        tasks.append(f"- **時系列データの前処理**: `{time_col}`列をdatetime型に変換してください。")
        if ts_features: tasks.append(f"- **時系列特徴量の作成**: 以下の特徴量を作成してください。\n  - " + "\n  - ".join(ts_features))
    else:
        # (従来の分類・回帰の指示)
        if source_type == "Kaggle形式": tasks.append("- **データ結合**: `df_train`と`df_test`を一度結合し、共通の前処理を実装してください。")
        tasks.append("- **カテゴリカル変数の処理**: **各カテゴリカル変数のユニーク数を調査**し、その数に応じてエンコーディングを適切に使い分けてください。")

    tasks.append("- **欠損値処理**: 欠損値の有無を確認し、もし存在すれば適切な方法で処理してください。")
    if include_scaling:
        tasks.append("- **特徴量スケーリング**: `StandardScaler`などを用いて、数値特徴量のスケールを揃える処理を実装してください。")

    prompt.append("\n".join(tasks))

    # ... (以降のStep 4, 5のロジックは、problem_type_full を使うように調整) ...
    # (煩雑になるため、ここでは主要な変更点のみを示しています)
    
    prompt.append("\n---\n以上の内容で、Pythonコードを生成してください。")
    return "\n".join(prompt)


# --------------------------------------------------------------------------
# Streamlit UI部 (✨✨ この部分を改善 ✨✨)
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
        
        ### --- 改善点: UIの選択肢を再構成 --- ###
        problem_type = st.radio("問題の種類は？", ["回帰", "分類", "時系列"], horizontal=True, key="problem_type")
        
        ts_task_type = ""
        if problem_type == "時系列":
            ts_task_type = st.radio(
                "時系列タスクを選択してください",
                ["予測 (Forecasting)", "分類 (Classification)"],
                horizontal=True, key="ts_task_type"
            )

    with st.expander("2. データソース", expanded=True):
        source_type = st.radio("データの形式は？", ["単一ファイル", "Kaggle形式"], horizontal=True, key="source_type")
        # (以下、変更なし)
        st.caption("Google Drive内のファイルパスを入力してください。")
        if source_type == "Kaggle形式":
            train_path = st.text_input("学習データ (train.csv)", "/content/drive/MyDrive/kaggle/train.csv")
            test_path = st.text_input("テストデータ (test.csv)", "/content/drive/MyDrive/kaggle/test.csv")
            submit_path = st.text_input("提出用サンプル", "/content/drive/MyDrive/kaggle/sample_submission.csv")
            single_path = ""
        else:
            single_path = st.text_input("CSVファイルのパス", "/content/drive/MyDrive/data/my_data.csv")
            train_path, test_path, submit_path = "", "", ""

    with st.expander("3. データの詳細", expanded=True):
        target_col_default = "y" if problem_type == "時系列" else "target"
        target_col = st.text_input("目的変数の列名", target_col_default, key="target_col")
        id_col = st.text_input("ID/識別子の列名", "id", key="id_col")
        
        time_col, ts_features = "", []
        # 時系列予測の場合のみ、特徴量作成の選択肢を表示
        if problem_type == "時系列" and ts_task_type == "予測 (Forecasting)":
            time_col = st.text_input("時系列カラムの列名", "datetime", key="time_col")
            ts_features = st.multiselect(
                "作成したい時系列特徴量",
                ["時間ベースの特徴量", "ラグ特徴量", "移動平均特徴量"],
                default=["時間ベースの特徴量", "ラグ特徴量", "移動平均特徴量"],
                key="ts_features"
            )
        elif problem_type == "時系列": # 予測・分類共通
             time_col = st.text_input("時系列カラムの列名", "timestamp", key="time_col")


    # (Expander 4, 5, 6 は主要なロジックに変更なし)
    with st.expander("4. データ概要", expanded=False):
        # ...
        data_context = st.session_state.data_context
    with st.expander("5. モデルと分析手法", expanded=True):
        # ...
        models = []
        use_ensemble, tune_hyperparams, use_feature_selection = False, False, False
        n_top_features = 50
        include_scaling, include_corr = True, True
    with st.expander("6. 可視化と保存先", expanded=True):
        # ...
        graphs = []
        save_dir_path = "/content/drive/MyDrive/results/"


# --- メイン画面 ---
st.header("✅ 生成されたプロンプト")
# (以下、プロンプト生成の実行部分は変更なし)
if all([target_col, data_context]):
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
    st.warning("サイドバーで必須項目（特に「目的変数」、「データ概要」など）を入力・選択してください。")
