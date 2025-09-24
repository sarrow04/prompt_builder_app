import streamlit as st

def generate_prompt(
    problem_type, source_type,
    train_path, test_path, submit_path, single_path,
    target_col, id_col, time_col,
    models, use_ensemble, tune_hyperparams,
    ts_features, include_stats, include_corr, graphs,
    # --- ▼▼▼ 保存先の引数を追加 ▼▼▼
    save_dir_path
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
        "!pip install japanize-matplotlib lightgbm shap holidays -q",
        "\n### 主要ライブラリのインポート",
        "# (pandas, numpy, lightgbm, sklearn, matplotlib, seaborn, shap, holidays, os などをインポートしてください)",
        "\n### Google Driveのマウント",
        "from google.colab import drive",
        "drive.mount('/content/drive')",
        "\n### ファイルパスと保存先の設定", # ### ▼▼▼ タイトル修正 ▼▼▼
    ]

    if source_type == "Kaggle形式":
        prompt.append(f"train_data_path = '{train_path}'")
        prompt.append(f"test_data_path = '{test_path}'")
        prompt.append(f"submit_template_path = '{submit_path}'")
    else: # 単一ファイル
        path_var = single_path
        prompt.append(f"file_path = '{path_var}'")
    
    # ### ▼▼▼ 保存先パスの変数を追加 ▼▼▼
    prompt.append(f"save_folder_path = '{save_dir_path}'")


    prompt.append("\n### データの読み込み")
    if source_type == "Kaggle形式":
        prompt.append("df_train = pd.read_csv(train_data_path)")
        prompt.append("df_test = pd.read_csv(test_data_path)")
        data_info_target = "`df_train`と`df_test`"
    else:
        prompt.append("df = pd.read_csv(file_path)")
        data_info_target = "`df`"

    # --- Step 2: データ概要の把握 ---
    prompt.extend([
        "\n# ==================================",
        "# Step 2: データ概要の把握",
        "# ==================================",
    ])
    if include_stats:
        prompt.append(f"読み込んだデータ({data_info_target})の概要を把握するため、`.info()`と`.head()`、`.describe(include='all')`を実行してください。")

    # --- Step 3: モデル構築と評価 ---
    prompt.extend([
        "\n# ==================================",
        "# Step 3: モデル構築と評価",
        "# ==================================",
        "### 実行してほしいタスク:",
    ])
    
    # ### ▼▼▼ 保存先ディレクトリ作成タスクを追加 ▼▼▼
    tasks = [f"- **保存先ディレクトリの作成**: `os.makedirs(save_folder_path, exist_ok=True)` を実行してください。"]
    
    # --- 前処理タスク ---
    if problem_type == "時系列予測":
        task_str = f"- **時系列データの前処理**: `{time_col}`列をdatetime型に変換し、インデックスに設定してください。"
        if source_type == "Kaggle形式":
            task_str = f"- **時系列データの前処理**: `df_train`と`df_test`を結合し、`{time_col}`列をdatetime型に変換して共通の時系列特徴量を作成してください。"
        tasks.append(task_str)

        if ts_features:
            feature_tasks = "\n- **時系列特徴量の作成**: 以下の特徴量を作成してください。\n"
            for feature in ts_features:
                feature_tasks += f"  - {feature}\n"
            tasks.append(feature_tasks)
    else:
        task_str = "- **前処理**: カテゴリカル変数のワンホットエンコーディング、欠損値の平均値補完など、基本的な前処理を実装してください。"
        if source_type == "Kaggle形式":
            task_str = "- **前処理**: `df_train`と`df_test`を一度結合し、共通の前処理（カテゴリ変数のワンホットエンコーディング、欠損値の平均値補完など）を実装してください。その後、再度`train`と`test`に分割してください。"
        tasks.append(task_str)


    if include_corr:
        tasks.append("- **相関分析**: 前処理後の特徴量間の相関行列を計算し、ヒートマップで可視化してください。")

    # --- モデル学習タスク ---
    model_str = "、".join(models)
    tasks.append(f"- **モデル学習**: `{model_str}` を使ってモデルを学習させてください。")
    
    if tune_hyperparams:
        cv_method = "TimeSeriesSplitを使ったクロスバリデーション" if problem_type == "時系列予測" else "通常のクロスバリデーション(cv=5)"
        tasks.append(f"- **ハイパーパラメータチューニング**: GridSearchCVを使い、`{models[0]}`の主要なハイパーパラメータをチューニングしてください。({cv_method})")
        
    if use_ensemble and len(models) > 1:
        tasks.append("- **アンサンブル**: 学習させた複数のモデルの予測値を平均し、アンサンブル予測を行ってください。")

    # --- モデル評価タスク ---
    evaluation_tasks = "\n- **モデル評価**: "
    # (省略: 前回のコードと同じ)
    tasks.append(evaluation_tasks)

    # --- 可視化タスク ---
    graph_tasks = "\n- **可視化**: 以下のグラフを生成し、`save_folder_path`に保存してください。\n" # ### ▼▼▼ 保存の指示を追加 ▼▼▼
    if "特徴量の重要度 (SHAP)" not in graphs:
        graphs.insert(0, "特徴量の重要度 (SHAP)")
    for graph in graphs:
        graph_tasks += f"  - {graph}\n"
    tasks.append(graph_tasks)

    # --- 出力タスク ---
    if output_format == "Kaggle形式":
        # ### ▼▼▼ 保存先パスを使用するように修正 ▼▼▼
        tasks.append(f"- **提出ファイルの作成**: `sample_submission.csv`の形式に合わせて、テストデータの予測結果を`os.path.join(save_folder_path, 'submission.csv')`として出力してください。ID列は`{id_col}`です。")
    elif problem_type == "時系列予測":
        tasks.append("- **未来予測**: 学習したモデルを使い、未来30期間の予測値を算出し、実績値と合わせてグラフにプロットしてください。")

    prompt.append("\n".join(tasks))
    prompt.append("\n---\n以上の内容で、Pythonコードを生成してください。")
    
    return "\n".join(prompt)


# --- Streamlit アプリのUI設定 ---
st.set_page_config(page_title="🤖 AIプロンプトビルダー", layout="wide")
st.title("🤖 AIプロンプトビルダー for Data Analysis")
st.write("データ分析のタスクをAIに依頼するための、完璧なプロンプトを自動生成します。")

# --- UI要素の初期化 ---
train_path, test_path, submit_path, single_path, time_series_path = None, None, None, None, None
id_col, time_col, ts_features = None, None, None
use_ensemble = False
graphs = []
save_dir_path = ""

with st.sidebar:
    st.header("⚙️ 設定項目")
    
    problem_type = st.radio("1. 分析の目的を選択", ["分類", "回帰", "時系列予測"], horizontal=True)

    st.subheader("2. データソースを選択")
    source_type = st.radio("データの形式", ["単一ファイル", "Kaggle形式"], horizontal=True)

    if source_type == "Kaggle形式":
        st.write("Google Drive内のファイルパスを入力してください。")
        train_path = st.text_input("学習データ (train.csv) のパス", "/content/drive/MyDrive/kaggle/train.csv")
        test_path = st.text_input("テストデータ (test.csv) のパス", "/content/drive/MyDrive/kaggle/test.csv")
        submit_path = st.text_input("提出用サンプル (sample_submission.csv) のパス", "/content/drive/MyDrive/kaggle/sample_submission.csv")
    else:
        st.write("Google Drive内のファイルパスを入力してください。")
        single_path = st.text_input("CSVファイルのパス", "/content/drive/MyDrive/data/my_data.csv")
    
    st.subheader("3. データの情報を入力")
    target_col = st.text_input("目的変数の列名", "y" if problem_type == "時系列予測" else "survived")
    
    if problem_type == "時系列予測":
        time_col = st.text_input("時系列カラム（日付/時刻）の列名", "ds")
    
    if source_type == "Kaggle形式":
        id_col = st.text_input("ID/識別子の列名", "id")
    elif problem_type != "時系列予測":
        id_col = st.text_input("ID/識別子の列名 (任意)", "id")
        
    if problem_type == "時系列予測":
        st.subheader("4. 時系列特徴量を選択")
        ts_features = st.multiselect("作成したい時系列特徴量", ["時間ベースの特徴量 (年, 月, 曜日など)", "ラグ特徴量", "移動平均特徴量", "祝日特徴量"], default=["時間ベースの特徴量 (年, 月, 曜日など)", "ラグ特徴量", "移動平均特徴量"])
    
    st.subheader("5. モデル戦略を選択")
    default_models = ["LightGBM"] if problem_type == "時系列予測" else ["LightGBM", "ロジスティック回帰/線形回帰"]
    models = st.multiselect("使用したいモデル", ["LightGBM", "ロジスティック回帰/線形回帰", "ランダムフォレスト", "XGBoost", "ARIMA", "Prophet"], default=default_models)
    if problem_type != "時系列予測":
        use_ensemble = st.checkbox("アンサンブル学習を行う", value=True)
    tune_hyperparams = st.checkbox("ハイパーパラメータチューニングを行う", value=True)

    st.subheader("6. 分析と可視化の項目を選択")
    include_stats = st.checkbox("データ概要の確認", value=True)
    include_corr = st.checkbox("相関ヒートマップの作成", value=True)
    
    # グラフオプション
    if problem_type == "分類":
        graph_options = ["目的変数の分布 (カウントプロット)", "特徴量の重要度 (SHAP)", "混同行列"]
    elif problem_type == "回帰":
        graph_options = ["目的変数の分布 (ヒストグラム)", "特徴量の重要度 (SHAP)", "実績値 vs 予測値プロット"]
    else: 
        graph_options = ["時系列グラフのプロット", "時系列分解図 (トレンド, 季節性)", "ACF/PACFプロット", "特徴量の重要度 (SHAP)"]
    graphs = st.multiselect("作成したいグラフの種類（SHAPは必須）", graph_options, default=graph_options)

    # ### ▼▼▼ ここからが変更・追加部分です ▼▼▼
    st.subheader("7. 保存先の設定")
    save_dir_path = st.text_input("グラフや提出ファイルの保存先フォルダ", "/content/drive/MyDrive/results/")
    # ### ▲▲▲ ここまでが変更・追加部分です ▲▲▲

# --- メイン画面に生成されたプロンプトを表示 ---
st.header("出力されたプロンプト")
st.info("以下のテキストをコピーして、AIアシスタントに貼り付けてください。")

if all([problem_type, target_col, models, source_type]):
    generated_prompt_text = generate_prompt(
        problem_type, source_type,
        train_path, test_path, submit_path, single_path,
        target_col, id_col, time_col,
        models, use_ensemble, tune_hyperparams,
        ts_features, include_stats, include_corr, graphs,
        save_dir_path # ### ▼▼▼ 保存先パスを関数に渡す ▼▼▼
    )
    st.text_area("生成されたプロンプト", generated_prompt_text, height=600)
else:
    st.warning("サイドバーで必須項目を設定してください。")
