import streamlit as st

def generate_prompt(
    problem_type, source_type,
    train_path, test_path, submit_path, single_path, time_series_path,
    target_col, id_col, time_col,
    models, use_ensemble, tune_hyperparams,
    ts_features, include_stats, include_corr, graphs
    ):
    """ユーザーの選択に基づいてAIへのプロンプトを生成する関数"""
    output_format = "Kaggle形式" if source_type == "Kaggle形式" else "一般的な分析・レポート"
    if problem_type == "時系列予測" and source_type != "Kaggle形式":
        output_format = "時系列予測レポート"

    # --- Step 1: 環境設定とデータ読み込み ---
    prompt = [
        "### 依頼内容：機械学習モデルの構築と評価",
        "これからGoogle Colab環境で実行する、データ分析のPythonコードをステップ・バイ・ステップで生成してください。",
        "\n# ==================================",
        "# Step 1: 環境設定とデータ読み込み",
        "# ==================================",
        "### ライブラリのインストール",
        "!pip install japanize-matplotlib lightgbm shap -q",
        "\n### 主要ライブラリのインポート",
        "# (pandas, numpy, lightgbm, sklearn, matplotlib, seaborn, shap などをインポートしてください)",
        "\n### Google Driveのマウント",
        "from google.colab import drive",
        "drive.mount('/content/drive')",
        "\n### ファイルパスの設定",
    ]

    if source_type == "Kaggle形式":
        prompt.append(f"train_data_path = '{train_path}'")
        prompt.append(f"test_data_path = '{test_path}'")
        prompt.append(f"submit_template_path = '{submit_path}'")
        prompt.append("\n### データの読み込み")
        prompt.append("df_train = pd.read_csv(train_data_path)")
        prompt.append("df_test = pd.read_csv(test_data_path)")
        data_info_target = "`df_train`と`df_test`"
    else: # 単一ファイル
        path_var = single_path
        prompt.append(f"file_path = '{path_var}'")
        prompt.append("\n### データの読み込み")
        prompt.append("df = pd.read_csv(file_path)")
        data_info_target = "`df`"

    # (以降のプロンプト生成ロジックは、前回のコードをベースに微修正)
    # ... (省略) ...
    #
    # 例えば、時系列+Kaggleの場合のタスクは以下のようになります
    if problem_type == "時系列予測" and source_type == "Kaggle形式":
        tasks = [
            f"- **時系列データの前処理**: `df_train`と`df_test`を結合し、`{time_col}`列をdatetime型に変換して共通の時系列特徴量を作成してください。",
            # ...
            "- **提出ファイルの作成**: ...",
        ]
        # (以下、他のタスクも同様に追加)
    
    # ここに前回の回答のプロンプト生成ロジックの大部分が入ります
    # 説明を簡潔にするため、UI部分の修正に焦点を当てます

    # (仮のプロンプトを返す)
    # 実際には前回の回答のプロンプト生成ロジックをここに記述します。
    final_prompt = "\n".join(prompt) + "\n\n(ここに動的に生成されたタスクリストが入ります)"
    return final_prompt

# --- Streamlit アプリのUI設定 ---
st.set_page_config(page_title="🤖 AIプロンプトビルダー", layout="wide")
st.title("🤖 AIプロンプトビルダー for Data Analysis")
st.write("データ分析のタスクをAIに依頼するための、完璧なプロンプトを自動生成します。")

# --- UI要素の初期化 ---
train_path, test_path, submit_path, single_path, time_series_path = None, None, None, None, None
id_col, time_col, ts_features = None, None, None
use_ensemble = False

with st.sidebar:
    st.header("⚙️ 設定項目")
    
    problem_type = st.radio("1. 分析の目的を選択", ["分類", "回帰", "時系列予測"], horizontal=True)

    # ### ▼▼▼ ここからが変更・追加部分です ▼▼▼
    st.subheader("2. データソースを選択")
    # 時系列予測でもKaggle形式を選べるように修正
    source_type = st.radio("データの形式", ["単一ファイル", "Kaggle形式"], horizontal=True)

    if source_type == "Kaggle形式":
        st.write("Google Drive内のファイルパスを入力してください。")
        train_path = st.text_input("学習データ (train.csv) のパス", "/content/drive/MyDrive/kaggle/train.csv")
        test_path = st.text_input("テストデータ (test.csv) のパス", "/content/drive/MyDrive/kaggle/test.csv")
        submit_path = st.text_input("提出用サンプル (sample_submission.csv) のパス", "/content/drive/MyDrive/kaggle/sample_submission.csv")
    else: # 単一ファイル
        st.write("Google Drive内のファイルパスを入力してください。")
        single_path = st.text_input("CSVファイルのパス", "/content/drive/MyDrive/data/my_data.csv")
    # ### ▲▲▲ ここまでが変更・追加部分です ▲▲▲
    
    st.subheader("3. データの情報を入力")
    target_col = st.text_input("目的変数の列名", "y" if problem_type == "時系列予測" else "survived")
    
    if problem_type == "時系列予測":
        time_col = st.text_input("時系列カラム（日付/時刻）の列名", "ds")
    
    # Kaggle形式の場合はID列が必須になることが多い
    if source_type == "Kaggle形式":
        id_col = st.text_input("ID/識別子の列名", "id")
    elif problem_type != "時系列予測":
        id_col = st.text_input("ID/識別子の列名 (任意)", "id")
        
    if problem_type == "時系列予測":
        st.subheader("4. 時系列特徴量を選択")
        ts_features = st.multiselect(
            "作成したい時系列特徴量",
            ["時間ベースの特徴量 (年, 月, 曜日など)", "ラグ特徴量", "移動平均特徴量", "祝日特徴量"],
            default=["時間ベースの特徴量 (年, 月, 曜日など)", "ラグ特徴量", "移動平均特徴量"]
        )
    
    # (以降のUI設定は前回のコードと同様)
    # ...
    models = ["LightGBM"] # 仮
    tune_hyperparams = True # 仮
    include_stats = True # 仮
    include_corr = True # 仮
    graphs = [] # 仮


# --- メイン画面に生成されたプロンプトを表示 ---
st.header("出力されたプロンプト")
st.info("以下のテキストをコピーして、AIアシスタントに貼り付けてください。")

if all([problem_type, target_col, models]):
    # (UIから取得した変数をgenerate_prompt関数に渡す)
    # 説明を簡潔にするため、表示部分は省略します
    st.text_area("生成されたプロンプト", "（ここに選択に応じたプロンプトが表示されます）", height=600)
