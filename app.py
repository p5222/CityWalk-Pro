import streamlit as st
import pandas as pd
import joblib
import os
import json
from datetime import datetime, timedelta
from openai import OpenAI
from prophet import Prophet
import plotly.express as px

# ==========================================
# 1. 系统配置
# ==========================================
st.set_page_config(page_title="CityWalk Pro 全国版", page_icon="🌏", layout="wide")

# ==========================================
# 0. 配置大模型 (使用硅基流动免费额度)
# ==========================================
# 填入你在硅基流动申请的 Key (sk-开头)
API_KEY = "sk-jaabjqopkduryfbotghlprjmpsadfhszpzcfspnmamarpdhb"
BASE_URL = "https://api.siliconflow.cn/v1"

# 注意：硅基流动的 DeepSeek 模型名字比较长，别写错
MODEL_NAME = "deepseek-ai/DeepSeek-V3"

client = OpenAI(api_key=API_KEY, base_url=BASE_URL)



# ==========================================
# 2. 动态加载服务
# ==========================================
def get_available_cities():
    """扫描 models 文件夹下的城市列表"""
    base_dir = "city_models"
    if not os.path.exists(base_dir):
        return []
    # 获取文件夹名称作为城市名
    cities = [d for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d))]
    return cities


@st.cache_resource
def load_city_models(city_name):
    """只加载选中城市的模型"""
    models = {}
    city_dir = os.path.join("city_models", city_name)
    if not os.path.exists(city_dir):
        return {}
    for filename in os.listdir(city_dir):
        if filename.endswith('.pkl'):
            spot_name = filename.replace('.pkl', '')
            try:
                models[spot_name] = joblib.load(os.path.join(city_dir, filename))
            except:
                pass
    return models


def predict_city_traffic(city_models, target_time):
    """预测该城市所有景点的客流"""
    results = []
    future_df = pd.DataFrame({'ds': [target_time]})

    for spot, model in city_models.items():
        forecast = model.predict(future_df)
        flow = int(forecast['yhat'].values[0])
        flow = max(0, flow)

        # --- 修复 1: 调整阈值，让颜色更丰富 ---
        # 之前的阈值太高了，导致全是绿色。现在调低一点。
        if flow < 200:
            status = "舒适 🟢"
            color_val = "green"
        elif flow < 400:
            status = "适中 🟡"
            color_val = "yellow"
        else:
            status = "拥挤 🔴"
            color_val = "red"

        results.append({
            "景点": spot,
            "预计客流": flow,
            "状态": status,
            "Color": color_val  # 用于排序或绘图
        })

    if not results:
        return pd.DataFrame(columns=["景点", "预计客流", "状态"])

    # 按客流从高到低排序
    return pd.DataFrame(results).sort_values(by="预计客流", ascending=False)


def call_llm_rag(user_query, city_name, traffic_data):
    """RAG: 注入当前城市的客流数据"""
    # 简化上下文，只传前10个，防止Token过多
    data_context = traffic_data[['景点', '预计客流', '状态']].head(10).to_string(index=False)

    system_prompt = f"""
    你是一个智能导游，当前城市：【{city_name}】。

    【实时客流数据】
    {data_context}

    【任务】
    1. 为用户规划路线，必须基于数据。
    2. 优先推荐“舒适 🟢”的景点，避开“拥挤 🔴”。
    3. 输出格式清晰，可以使用 Markdown 列表。
    """

    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_query}
            ],
            stream=True
        )
        return response
    except Exception as e:
        return str(e)


# ==========================================
# 3. 前端界面 (修复版 - 移除不稳定 HTML)
# ==========================================
# CSS 仅用于美化原生组件，不改变结构
st.markdown("""
<style>
    /* 调整侧边栏背景 */
    section[data-testid="stSidebar"] {
        background-color: #f8f9fa;
    }
    /* 隐藏页脚 */
    footer {visibility: hidden;}
    header {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

# --- 侧边栏：城市与监控 ---
with st.sidebar:
    st.title("🌏 城市指挥中心")

    available_cities = get_available_cities()
    if not available_cities:
        st.error("❌ 请先运行 data_engine.py 生成数据")
        current_city = None
    else:
        # 默认选深圳
        default_idx = available_cities.index("深圳") if "深圳" in available_cities else 0
        current_city = st.selectbox("📍 切换城市", available_cities, index=default_idx)

    st.divider()

    # 实时监控图表
    if current_city:
        st.markdown(f"### 📊 {current_city}实时热力")
        city_models = load_city_models(current_city)

        if city_models:
            df_traffic = predict_city_traffic(city_models, datetime.now())

            # 颜色映射
            color_map = {
                "舒适 🟢": "#2ecc71",  # 绿
                "适中 🟡": "#f1c40f",  # 黄
                "拥挤 🔴": "#e74c3c"  # 红
            }

            # 使用 Plotly 画图
            fig = px.bar(
                df_traffic,
                x='预计客流',
                y='景点',
                orientation='h',
                color='状态',
                color_discrete_map=color_map,
                text='预计客流',
                height=500
            )
            fig.update_layout(
                xaxis_title="",
                yaxis_title="",
                showlegend=False,
                yaxis={'categoryorder': 'total ascending'}  # 自动排序
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("数据加载中...")

# --- 主界面：聊天区 (使用原生组件，不再报错) ---
if current_city:
    st.title(f"🚀 CityWalk Pro · {current_city}站")
    st.caption("基于运营商核心数据 | RAG 检索增强生成")

    # 初始化历史
    if "messages" not in st.session_state:
        st.session_state.messages = []
    # 切换城市时清空历史，防止串台
    if "last_city" not in st.session_state or st.session_state.last_city != current_city:
        st.session_state.messages = []
        st.session_state.messages.append({
            "role": "assistant",
            "content": f"👋 欢迎来到 **{current_city}**！\n\n我是你的 AI 伴游，我已经获取了全城景点的实时客流数据。\n你可以问我：\n- *“帮我规划一条人少的半日游路线”*\n- *“现在去哪里玩比较舒服？”*"
        })
        st.session_state.last_city = current_city

    # 1. 渲染历史消息 (使用 st.chat_message 稳定性 MAX)
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # 2. 处理新输入
    if prompt := st.chat_input(f"在 {current_city} 怎么玩？"):
        # 用户消息
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # AI 回复
        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            full_response = ""

            # 调用 LLM
            stream = call_llm_rag(prompt, current_city, df_traffic)

            if isinstance(stream, str):
                st.error(f"出错啦: {stream}")
            else:
                for chunk in stream:
                    if chunk.choices[0].delta.content:
                        content = chunk.choices[0].delta.content
                        full_response += content
                        message_placeholder.markdown(full_response + "▌")

                message_placeholder.markdown(full_response)

        # 存入历史
        st.session_state.messages.append({"role": "assistant", "content": full_response})

else:
    st.info("👈 请先在左侧选择城市")