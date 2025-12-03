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


# 使用 cache 提升性能，当 city 改变时重新加载
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

        if flow < 400:
            status = "舒适 🟢"
        elif flow < 800:
            status = "适中 🟡"
        else:
            status = "拥挤 🔴"

        results.append({"景点": spot, "预计客流": flow, "状态": status})

    if not results:
        return pd.DataFrame(columns=["景点", "预计客流", "状态"])

    return pd.DataFrame(results).sort_values(by="预计客流")


def call_llm_rag(user_query, city_name, traffic_data):
    """RAG: 注入当前城市的客流数据"""
    data_context = traffic_data.to_string(index=False)

    system_prompt = f"""
    你是一个智能导游，当前用户所在的城市是【{city_name}】。

    【该城市实时客流监测】
    {data_context}

    【任务】
    1. 根据客流数据，为用户规划在【{city_name}】的游玩路线。
    2. 必须优先推荐“舒适”状态的景点。
    3. 如果用户问到其他城市，请礼貌提醒先切换城市。
    4. 结合该城市的文化特色（如西安的历史、重庆的魔幻）进行讲解。
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
        return f"AI 服务异常: {e}"


# ==========================================
# 3. 前端界面
# ==========================================
# CSS 美化
st.markdown("""
<style>
    .stApp {background-color: #ffffff;}
    .css-1d391kg {padding-top: 1rem;} 
    /* 侧边栏优化 */
    section[data-testid="stSidebar"] {
        background-color: #f7f9fc;
        border-right: 1px solid #e3e6f0;
    }
</style>
""", unsafe_allow_html=True)

# --- 侧边栏：城市选择与监控 ---
with st.sidebar:
    st.title("🌏 城市指挥中心")

    # 1. 城市选择器
    available_cities = get_available_cities()
    if not available_cities:
        st.error("❌ 未找到数据，请运行 data_engine.py")
        current_city = None
    else:
        # 默认选北京，如果没有则选第一个
        default_idx = available_cities.index("北京") if "北京" in available_cities else 0
        current_city = st.selectbox("📍 当前城市", available_cities, index=default_idx)

    st.divider()

    # 2. 实时监控
    if current_city:
        st.markdown(f"### 📊 {current_city}实时热力")
        target_time = datetime.now()  # 默认为当前时间

        # 加载模型并预测
        city_models = load_city_models(current_city)
        if city_models:
            df_traffic = predict_city_traffic(city_models, target_time)

            # 展示图表
            fig = px.bar(df_traffic, x='预计客流', y='景点', orientation='h',
                         color='状态',
                         color_discrete_map={"舒适 🟢": "#2ecc71", "适中 🟡": "#f1c40f", "拥挤 🔴": "#e74c3c"},
                         height=400)
            fig.update_layout(xaxis_title="", yaxis_title="", showlegend=False)
            st.plotly_chart(fig, use_container_width=True)

            # 展示详细数据表
            with st.expander("查看详细数据"):
                st.dataframe(df_traffic, hide_index=True)
        else:
            st.warning("模型加载中...")

# --- 主界面 ---
if current_city:
    st.title(f"🚀 CityWalk Pro · {current_city}站")
    st.caption(f"基于运营商核心数据 | 覆盖全国 {len(available_cities)} 个热门城市")

    # 初始化历史记录 (切换城市时清空历史，避免上下文混乱)
    if "last_city" not in st.session_state or st.session_state.last_city != current_city:
        st.session_state.messages = [{"role": "assistant",
                                      "content": f"欢迎来到{current_city}！您可以问我：\n\n“{current_city}有哪些人少好玩的地方？”\n“帮我规划一条{current_city}的半日游路线。”"}]
        st.session_state.last_city = current_city

    # 渲染聊天
    for msg in st.session_state.messages:
        st.chat_message(msg["role"]).write(msg["content"])

    # 输入框
    if prompt := st.chat_input(f"问问 {current_city} 怎么玩..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        st.chat_message("user").write(prompt)

        with st.chat_message("assistant"):
            placeholder = st.empty()
            full_resp = ""

            # 传入城市名和数据
            stream = call_llm_rag(prompt, current_city, df_traffic)

            if isinstance(stream, str):
                placeholder.error(stream)
            else:
                for chunk in stream:
                    if chunk.choices[0].delta.content:
                        full_resp += chunk.choices[0].delta.content
                        placeholder.markdown(full_resp + "▌")
                placeholder.markdown(full_resp)

        st.session_state.messages.append({"role": "assistant", "content": full_resp})
else:
    st.info("👈 请在左侧选择一个城市开始")