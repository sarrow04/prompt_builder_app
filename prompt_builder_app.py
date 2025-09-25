import streamlit as st
import pandas as pd
from io import StringIO
import numpy as np

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

    # Step 1
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

    # Step 2
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

    # Step 3
    prompt.extend([
        "\n# ==================================",
        f"# Step 3: {analysis_goal}のためのモデル構築と評価",
        "# ==================================",
        "### 実行してほしいタスク:",
    ])
    
    tasks = [f"- **保存先ディレクトリの作成**: `os.makedirs(save_folder_path, exist_ok=True)` を実行してください。"]
    
    # 前処理タスク
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

    # モデル学習タスク
    if analysis_goal == "要因分析":
        tasks.append("- **モデル学習**: **解釈性の高い**モデル（ロジスティック回帰/線形回帰, 決定木など）を優先して使用してください。")
        tasks.append(f"- **要因分析**: 学習したモデルの係数やSHAP値を使い、「どの特徴量が `{target_col}` に正または負の影響を与えているか」を分析・考察してください。")
    else:
        model_str = "、".join(models)
        tasks.append(f"- **モデル学習**: `{model_str}` を使って、予測精度が最大になるようにモデルを学習させてください。")
        
        if tune_hyperparams and models:
            cv_method = "TimeSeriesSplitを使ったクロスバリデーション" if problem_type == "時系列予測" else "通常のクロスバリデーション(cv=5)"
            tasks.append(f"- **ハイパーパラメータチューニング**: GridSearchCVを使い、`{models[0]}`のモデルの予測精度をさらに向上させてください。({cv_method})")
        
        if use_ensemble and len(models) > 1:
            tasks.append("- **アンサンブル**: 学習させた複数のモデルの予測値を平均し、アンサンブル予測を行ってください。")

    # モデル評価タスク
    evaluation_tasks = "\n- **モデル評価**: "
    if problem_type == "分類":
        evaluation_tasks += "テストデータに対して、以下の指標を計算し、結果を報告してください。\n  - 混同行列 (Confusion Matrix)\n  - 正解率 (Accuracy), 適合率 (Precision), 再現率 (Recall), F1スコア\n  - AUCスコア"
    elif problem_type == "回帰":
        evaluation_tasks += "テストデータに対して、以下の指標を計算し、結果を報告してください。\n  - RMSE (Root Mean Squared Error)\n  - MAE (Mean Absolute Error)\n  - R2スコア (決定係数)"
    elif problem_type == "時系列予測":
        evaluation_tasks += "テストデータ（またはバックテスト区間）に対して、以下の指標を計算してください。\n  - RMSE (Root Mean Squared Error)\n  - MAE (Mean Absolute Error)\n  - MAPE (Mean Absolute Percentage Error)"
    tasks.append(evaluation_tasks)

    # 可視化タスク
    graph_tasks = "\n- **可視化**: 以下のグラフを生成し、`save_folder_path`に保存してください。\n"
    if "特徴量の重要度 (SHAP)" not in graphs:
        graphs.insert(0, "特徴量の重要度 (SHAP)")
    for graph in graphs:
        graph_tasks += f"  - {graph}\n"
    tasks.append(graph_tasks)

    # 出力タスク
    if output_format == "Kaggle形式":
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

if 'data_context' not in st.session_state:
    st.session_state.data_context = ""

with st.sidebar:
    st.header("⚙️ 設定項目")
    
    # ✨ FIX: 全てのUI変数を最初に初期化し、NameErrorを完全に防ぐ
    train_path, test_path, submit_path, single_path = "", "", "", ""
    id_col, time_col, ts_features = "", "", []
    use_ensemble, tune_hyperparams, include_corr = False, False, True

    st.subheader("1. 分析のゴールを選択")
    analysis_goal = st.radio("この分析の主な目的は？", ["予測モデルの構築", "要因分析"], horizontal=True)
    problem_type = st.radio("2. 分析の目的を選択", ["分類", "回帰", "時系列予測"], horizontal=True)

    st.subheader("3. データソース")
    source_type = st.radio("データの形式", ["単一ファイル", "Kaggle形式"], horizontal=True)
    
    st.write("Google Drive内のファイルパスを入力してください。")
    if source_type == "Kaggle形式":
        train_path = st.text_input("学習データ (train.csv) のパス", "/content/drive/MyDrive/kaggle/train.csv")
        test_path = st.text_input("テストデータ (test.csv) のパス", "/content/drive/MyDrive/kaggle/test.csv")
        submit_path = st.text_input("提出用サンプル (sample_submission.csv) のパス", "/content/drive/MyDrive/kaggle/sample_submission.csv")
    else: # 単一ファイル
        single_path = st.text_input("CSVファイルのパス", "/content/drive/MyDrive/data/my_data.csv")
    
    st.subheader("4. データの情報")
    target_col_default = "y" if problem_type == "時系列予測" else "survived"
    target_col = st.text_input("目的変数の列名", target_col_default)
    
    # ✨ FIX: ID列の入力UIをシンプルにし、全てのケースをカバー
    if source_type == "Kaggle形式":
        id_col = st.text_input("ID/識別子の列名", "id")
    
    if problem_type == "時系列予測":
        time_col = st.text_input("時系列カラム（日付/時刻）の列名", "ds")
    
    uploaded_file_for_context = st.file_uploader(
        "ここにCSVをアップロードして概要を自動生成", type=['csv'],
        help="分析対象のファイル（単一ファイルの場合）または学習データ（Kaggle形式の場合）をアップロードしてください。"
    )
    if uploaded_file_for_context:
        try:
            df_context = pd.read_csv(uploaded_file_for_context)
            buffer = StringIO()
            df_context.info(buf=buffer)
            info_str = buffer.getvalue()
            head_str = df_context.head().to_markdown()
            st.session_state.data_context = f"【df.info()】\n{info_str}\n\n【df.head()】\n{head_str}"
        except Exception as e:
            st.error(f"ファイル読み込みエラー: {e}")
    data_context = st.text_area("データ概要（自動入力）", value=st.session_state.data_context, height=200)

    st.subheader("5. モデル戦略")
    default_models = ["LightGBM"] if problem_type == "時系列予測" else ["LightGBM", "ロジスティック回帰/線形回帰"]
    models = st.multiselect("使用したいモデル", ["LightGBM", "ロジスティック回帰/線形回帰", "ランダムフォレスト", "XGBoost", "ARIMA", "Prophet"], default=default_models)
    
    if analysis_goal == "予測モデルの構築":
        tune_hyperparams = st.checkbox("ハイパーパラメータチューニングを行う", value=True)
        if problem_type != "時系列予測" and len(models) > 1:
            use_ensemble = st.checkbox("アンサンブル学習を行う", value=True)
        
    st.subheader("6. 分析と可視化")
    include_corr = st.checkbox("相関ヒートマップを作成", value=True)
    
    if problem_type == "時系列予測":
        graph_options = ["時系列グラフのプロット", "時系列分解図 (トレンド, 季節性)", "ACF/PACFプロット", "特徴量の重要度 (SHAP)"]
        ts_features = st.multiselect("作成したい時系列特徴量", ["時間ベースの特徴量 (年, 月, 曜日など)", "ラグ特徴量", "移動平均特徴量", "祝日特徴量"], default=["時間ベースの特徴量 (年, 月, 曜日など)", "ラグ特徴量", "移動平均特徴量"])
    elif problem_type == "分類":
        graph_options = ["目的変数の分布 (カウントプロット)", "特徴量の重要度 (SHAP)", "混同行列"]
    else: # 回帰
        graph_options = ["目的変数の分布 (ヒストグラム)", "特徴量の重要度 (SHAP)", "実績値 vs 予測値プロット"]
    
    graphs = st.multiselect("作成したいグラフの種類（SHAPは必須）", graph_options, default=graph_options)
    
    st.subheader("7. 保存先")
    save_dir_path = st.text_input("グラフや提出ファイルの保存先フォルダ", "/content/drive/MyDrive/results/")

# --- メイン画面 ---
st.header("✅ 生成されたプロンプト")
st.info("以下のテキストをコピーして、AIアシスタントに貼り付けてください。")

# ✨ FIX: `models`が空でないこともチェック条件に追加
if all([problem_type, target_col, source_type, data_context]) and models:
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
    st.warning("サイドバーで必須項目（ゴール、目的、データ概要、使用モデルなど）を設定・入力してください。")
