import streamlit as st
import pandas as pd
from io import StringIO

# (generate_prompt関数は前回の回答と同じなので、ここでは省略します)
# ...
def generate_prompt(
    problem_type, source_type, analysis_goal, data_context,
    train_path, test_path, submit_path, single_path,
    target_col, id_col, time_col,
    models, use_ensemble, tune_hyperparams,
    ts_features, include_corr, graphs,
    save_dir_path
    ):
    """(前回の回答と同じコードがここに入ります)"""
    # ...
    # この関数は変更なし
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

    prompt.extend([
        "\n# ==================================",
        f"# Step 3: {analysis_goal}のためのモデル構築と評価",
        "# ==================================",
        "### 実行してほしいタスク:",
    ])
    
    tasks = [f"- **保存先ディレクトリの作成**: `os.makedirs(save_folder_path, exist_ok=True)` を実行してください。"]
    
    # ...(以降のタスク生成ロジックは省略)...

    prompt.append("\n".join(tasks))
    prompt.append("\n---\n以上の内容で、Pythonコードを生成してください。")
    
    return "\n".join(prompt)


# --- Streamlit アプリのUI設定 ---
st.set_page_config(page_title="🤖 AIプロンプトビルダー", layout="wide")
st.title("🤖 AIプロンプトビルダー for Data Analysis")
st.write("データ分析のタスクをAIに依頼するための、完璧なプロンプトを自動生成します。")

# --- UI要素の初期化 (省略) ---
if 'data_context' not in st.session_state:
    st.session_state.data_context = ""


with st.sidebar:
    st.header("⚙️ 設定項目")
    
    st.subheader("1. 分析のゴールを選択")
    analysis_goal = st.radio("この分析の主な目的は？", ["予測モデルの構築", "要因分析"], horizontal=True)

    problem_type = st.radio("2. 分析の目的を選択", ["分類", "回帰", "時系列予測"], horizontal=True)

    st.subheader("3. データソース（プロンプト生成用）")
    source_type = st.radio("データの形式", ["単一ファイル", "Kaggle形式"], horizontal=True)

    if source_type == "Kaggle形式":
        st.write("Google Drive内のファイルパスを入力してください。")
        train_path = st.text_input("学習データ (train.csv) のパス", "/content/drive/MyDrive/kaggle/train.csv")
        test_path = st.text_input("テストデータ (test.csv) のパス", "/content/drive/MyDrive/kaggle/test.csv")
        submit_path = st.text_input("提出用サンプル (sample_submission.csv) のパス", "/content/drive/MyDrive/kaggle/sample_submission.csv")
    else: # 単一ファイル
        st.write("Google Drive内のファイルパスを入力してください。")
        single_path = st.text_input("CSVファイルのパス", "/content/drive/MyDrive/data/my_data.csv")

    st.subheader("4. データの情報を入力")
    target_col = st.text_input("目的変数の列名", "y" if problem_type == "時系列予測" else "survived")

    # ### ▼▼▼ ここからが変更・追加部分です ▼▼▼
    st.subheader("重要：データ概要の自動入力")
    uploaded_file_for_context = st.file_uploader(
        "ここにCSVをアップロードして概要を自動生成",
        type=['csv'],
        help="ここでアップロードしたファイルは、下の「データ概要」を生成するためだけに使われます。ファイルの中身は送信されません。"
    )

    # ファイルがアップロードされたら、infoとheadを生成してセッションステートに保存
    if uploaded_file_for_context is not None:
        try:
            df_context = pd.read_csv(uploaded_file_for_context)
            
            # df.info()の結果を文字列として取得
            buffer = StringIO()
            df_context.info(buf=buffer)
            info_str = buffer.getvalue()
            
            # df.head()の結果を文字列（Markdown形式）として取得
            head_str = df_context.head().to_markdown()
            
            # セッションステートに保存
            st.session_state.data_context = f"【df.info()】\n{info_str}\n\n【df.head()】\n{head_str}"
            st.success("データ概要を生成しました！")
        except Exception as e:
            st.error(f"ファイル読み込みエラー: {e}")

    data_context = st.text_area(
        "データ概要（自動入力）",
        value=st.session_state.data_context, # セッションステートから値を読み込む
        height=200,
        help="上のアップローダーを使うと、この欄に自動で入力されます。"
    )
    # ### ▲▲▲ ここまでが変更・追加部分です ▲▲▲

    # (以降のUI設定は前回のコードと同様)
    # ...
    # (仮の値を設定)
    models, use_ensemble, tune_hyperparams, ts_features, include_corr, graphs, save_dir_path = ["LightGBM"], False, False, [], False, [], ""


# --- メイン画面に生成されたプロンプトを表示 ---
st.header("出力されたプロンプト")
st.info("以下のテキストをコピーして、AIアシスタントに貼り付けてください。")

if all([problem_type, target_col, models, source_type, data_context]):
    generated_prompt_text = generate_prompt(
        problem_type, source_type, analysis_goal, data_context,
        train_path, test_path, submit_path, single_path,
        target_col, id_col, time_col,
        models, use_ensemble, tune_hyperparams,
        ts_features, include_corr, graphs,
        save_dir_path
    )
    st.text_area("生成されたプロンプト", generated_prompt_text, height=600)
else:
    st.warning("サイドバーで必須項目（ゴール、目的、データ概要など）を設定してください。")
