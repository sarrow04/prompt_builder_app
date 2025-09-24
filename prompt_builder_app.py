import streamlit as st

def generate_prompt(
    problem_type, source_type, analysis_goal, data_context,
    train_path, test_path, submit_path, single_path,
    target_col, id_col, time_col,
    models, use_ensemble, tune_hyperparams,
    ts_features, include_corr, graphs,
    save_dir_path
    ):
    """ユーザーの選択に基づいてAIへのプロンプトを生成する関数"""
    output_format = "Kaggle形式" if source_type == "Kaggle形式" else "一般的な分析・レポート"
    if problem_type == "時系列予測" and source_type != "Kaggle形式":
        output_format = "時系列予測レポート"

    # --- Step 1: 環境設定とデータ読み込み ---
    prompt = [
        f"### 依頼内容：機械学習を用いた「{problem_type}」の「{analysis_goal}」",
        "これからGoogle Colab環境で実行する、データ分析のPythonコードをステップ・バイ・ステップで生成してください。",
        "\n# ==================================",
        "# Step 1: 環境設定とデータ読み込み",
        "# ==================================",
        "### ライブラリのインストール",
        "!pip install japanize-matplotlib lightgbm shap holidays -q",
        "\n### 主要ライブラリのインポート",
        "# (pandas, numpy, lightgbm, sklearn, matplotlib, seaborn, shap, holidays, os などをインポートしてください)",
        "\n### Google Driveのマウント",
        "from google.colab import drive",
        "drive.mount('/content/drive')",
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

    # --- Step 2: データ概要の把握 ---
    prompt.extend([
        "\n# ==================================",
        "# Step 2: データ概要の把握",
        "# ==================================",
        "### データのカラム名とサンプル",
        "以下に、分析対象となるデータのカラム構成と先頭数行のサンプルを示します。これを元に分析を進めてください。",
        "--- データ概要 ---",
        data_context,
        "--- ここまで ---",
        f"\n今回の分析の目的変数は `{target_col}` です。"
    ])

    # --- Step 3: モデル構築と評価 ---
    prompt.extend([
        "\n# ==================================",
        f"# Step 3: {analysis_goal}のためのモデル構築と評価",
        "# ==================================",
        "### 実行してほしいタスク:",
    ])
    
    tasks = [f"- **保存先ディレクトリの作成**: `os.makedirs(save_folder_path, exist_ok=True)` を実行してください。"]
    
    # (省略: 前回のコードのタスク生成ロジックがここに入る)
    # ...
    # ### ▼▼▼ 分析ゴールに応じたタスクの微調整 ▼▼▼
    if analysis_goal == "要因分析":
        tasks.append("- **モデル学習**: **解釈性の高い**モデル（ロジスティック回帰/線形回帰, 決定木など）を優先して使用してください。")
        tasks.append(f"- **要因分析**: 学習したモデルの係数やSHAP値を使い、「どの特徴量が `{target_col}` に正または負の影響を与えているか」を分析・考察してください。")
    else: # 予測モデルの構築
        model_str = "、".join(models)
        tasks.append(f"- **モデル学習**: `{model_str}` を使って、予測精度が最大になるようにモデルを学習させてください。")
        if tune_hyperparams:
            tasks.append("- **ハイパーパラメータチューニング**: GridSearchCVを使い、モデルの予測精度をさらに向上させてください。")
    # (以下、評価指標や可視化、提出ファイルのタスクが続く)
    # ...

    prompt.append("\n".join(tasks))
    prompt.append("\n---\n以上の内容で、Pythonコードを生成してください。")
    
    return "\n".join(prompt)


# --- Streamlit アプリのUI設定 ---
st.set_page_config(page_title="🤖 AIプロンプトビルダー", layout="wide")
st.title("🤖 AIプロンプトビルダー for Data Analysis")
st.write("データ分析のタスクをAIに依頼するための、完璧なプロンプトを自動生成します。")

# --- UI要素の初期化 (省略) ---

with st.sidebar:
    st.header("⚙️ 設定項目")
    
    # ### ▼▼▼ 「分析のゴール」選択を追加 ▼▼▼
    st.subheader("1. 分析のゴールを選択")
    analysis_goal = st.radio("この分析の主な目的は？", ["予測モデルの構築", "要因分析"], horizontal=True)

    problem_type = st.radio("2. 分析の目的を選択", ["分類", "回帰", "時系列予測"], horizontal=True)

    # (中略: データソース選択)
    
    st.subheader("4. データの情報を入力")
    target_col = st.text_input("目的変数の列名", "y" if problem_type == "時系列予測" else "survived")
    
    # ### ▼▼▼ データ概要の貼り付け欄を追加 ▼▼▼
    st.subheader("重要：データ概要の貼り付け")
    data_context = st.text_area(
        "ここに `df.info()` と `df.head()` の結果を貼り付け",
        height=150,
        help="分析対象のDataFrameの概要をAIに正確に伝えるため、Colabなどで実行した結果をここに貼り付けてください。"
    )

    # (以降のUI設定は前回のコードと同様)
    # ...

# --- メイン画面に生成されたプロンプトを表示 ---
st.header("出力されたプロンプト")
st.info("以下のテキストをコピーして、AIアシスタントに貼り付けてください。")

# if all() の条件に data_context を追加
if all([problem_type, target_col, models, source_type, data_context]):
    # (generate_promptに関数を渡す)
    # ...
    pass
else:
    st.warning("サイドバーで必須項目（ゴール、目的、データ概要など）を設定してください。")

