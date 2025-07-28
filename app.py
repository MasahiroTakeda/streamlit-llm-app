import streamlit as st
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain.schema import SystemMessage, HumanMessage

# 環境変数を読み込み
load_dotenv()

# Streamlitページ設定
st.set_page_config(page_title="LLM Chat App", page_icon="🤖")

# タイトル
st.title("🤖 専門家LLMチャットアプリ")
st.write("専門家を選択して、質問を入力してください。選択した専門家として回答します。")

# LLMの初期化
@st.cache_resource
def initialize_llm():
    return ChatOpenAI(model_name="gpt-4o-mini", temperature=0)

llm = initialize_llm()

# 専門家の種類とシステムメッセージの定義
EXPERT_TYPES = {
    "医師": {
        "name": "医師",
        "system_message": "あなたは経験豊富な医師です。医学的な知識に基づいて、正確で分かりやすい回答を提供してください。ただし、具体的な診断や治療法については、必ず実際の医師に相談するよう伝えてください。"
    },
    "弁護士": {
        "name": "弁護士", 
        "system_message": "あなたは経験豊富な弁護士です。法的な問題について、法律に基づいた正確な情報を提供してください。ただし、具体的な法的アドバイスについては、実際の弁護士に相談するよう伝えてください。"
    },
    "プログラマー": {
        "name": "プログラマー",
        "system_message": "あなたは経験豊富なソフトウェアエンジニアです。プログラミングに関する質問に対して、実践的で効率的なソリューションを提供してください。コード例も含めて分かりやすく説明してください。"
    },
    "料理研究家": {
        "name": "料理研究家",
        "system_message": "あなたは経験豊富な料理研究家です。料理や食材に関する質問に対して、実践的で美味しいレシピや調理法を提供してください。栄養や食材の特性についても詳しく説明してください。"
    },
    "心理カウンセラー": {
        "name": "心理カウンセラー",
        "system_message": "あなたは経験豊富な心理カウンセラーです。心理的な悩みや人間関係の問題について、共感的で建設的なアドバイスを提供してください。深刻な心理的問題については、専門機関への相談を勧めてください。"
    }
}

# LLMに質問を送信して回答を取得する関数
def get_expert_response(user_input: str, expert_type: str) -> str:
    """
    指定された専門家として、ユーザーの質問に回答を生成する
    
    Args:
        user_input (str): ユーザーからの質問テキスト
        expert_type (str): 選択された専門家の種類
    
    Returns:
        str: LLMからの回答
    """
    try:
        # 選択された専門家のシステムメッセージを取得
        system_message = EXPERT_TYPES[expert_type]["system_message"]
        
        # メッセージを作成
        messages = [
            SystemMessage(content=system_message),
            HumanMessage(content=user_input),
        ]
        
        # LLMに送信して回答を取得
        result = llm(messages)
        return result.content
        
    except Exception as e:
        return f"エラーが発生しました: {str(e)}"

# 入力フォーム
with st.form("chat_form"):
    # 専門家選択のラジオボタン
    expert_choice = st.radio(
        "どの専門家に相談しますか？",
        options=list(EXPERT_TYPES.keys()),
        format_func=lambda x: EXPERT_TYPES[x]["name"],
        index=0
    )
    
    st.write(f"**選択された専門家**: {EXPERT_TYPES[expert_choice]['name']}")
    
    # 質問入力エリア
    user_input = st.text_area(
        "質問を入力してください:",
        placeholder="例: 健康について気になることがあります。",
        height=100
    )
    
    submitted = st.form_submit_button("送信")

# フォームが送信された場合の処理
if submitted and user_input:
    with st.spinner("回答を生成中..."):
        # 専門家として回答を生成
        response = get_expert_response(user_input, expert_choice)
        
        # 回答を表示
        st.success(f"{EXPERT_TYPES[expert_choice]['name']}からの回答:")
        st.write(response)

elif submitted and not user_input:
    st.warning("質問を入力してください。")