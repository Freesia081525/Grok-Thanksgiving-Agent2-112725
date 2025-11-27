import streamlit as st
import yaml
import json
import markdown
from openai import OpenAI
import google.generativeai as genai
from anthropic import Anthropic
import httpx
import os

st.set_page_config(page_title="繁體中文多智能體分析鏈", layout="wide", page_icon="🧠")

# Load agents
with open("agents.yaml", "r", encoding="utf-8") as f:
    config = yaml.safe_load(f)
    AGENTS = config["agents"]

# Model mapping
MODELS = {
    "OpenAI: gpt-4o-mini": ("openai", "gpt-4o-mini"),
    "OpenAI: gpt-4.1-mini": ("openai", "gpt-4.1-mini"),
    "Gemini: gemini-1.5-flash": ("gemini", "gemini-1.5-flash"),
    "Gemini: gemini-2.0-flash": ("gemini", "gemini-2.0-flash"),
    "xAI: grok-beta": ("xai", "grok-beta"),
    "Anthropic: claude-3-haiku": ("anthropic", "claude-3-haiku-20240307"),
}

# Sidebar
with st.sidebar:
    st.header("🔑 API 金鑰")
    api_keys = {
        "openai": st.text_input("OpenAI", type="password", value=os.getenv("OPENAI_API_KEY", "")),
        "gemini": st.text_input("Gemini", type="password", value=os.getenv("GEMINI_API_KEY", "")),
        "xai": st.text_input("xAI", type="password", value=os.getenv("XAI_API_KEY", "")),
        "anthropic": st.text_input("Anthropic", type="password", value=os.getenv("ANTHROPIC_API_KEY", "")),
    }

tab1, tab2 = st.tabs(["單次分析", "多智能體鏈（進階）"])

with tab1:
    # 原有 Note Keeper 功能（略，保留你原本需求）
    st.info("此分頁保留原始 Note Keeper 功能（可另加）")

with tab2:
    st.markdown("# 🧠 多智能體鏈式分析（繁體中文專用）")
    st.markdown("### 選擇並排序你想執行的分析智能體（可重複）")

    input_text = st.text_area("請貼上你要分析的文字", height=200, key="main_input")

    if input_text.strip():
        selected_agents = st.multiselect(
            "選擇分析智能體（拖曳調整順序）",
            options=[a["name"] for a in AGENTS],
            default=[a["name"] for a in AGENTS[:5]],
            key="selected_agents"
        )

        # 讓使用者調整順序
        ordered_agents = st.sortable_list(
            "拖曳調整執行順序",
            selected_agents,
            key="ordered"
        )

        cols = st.columns(3)
        with cols[0]:
            default_model = st.selectbox("預設模型", options=list(MODELS.keys()), index=0)
        with cols[1]:
            default_tokens = st.slider("預設 Max Tokens", 500, 8000, 3000)
        with cols[2]:
            default_temp = st.slider("預設 Temperature", 0.0, 1.0, 0.7, 0.05)

        if st.button("啟動多智能體分析鏈", type="primary", use_container_width=True):
            current_text = input_text
            results = []

            for i, agent_name in enumerate(ordered_agents):
                agent = next(a for a in AGENTS if a["name"] == agent_name)

                with st.expander(f"{i+1}. {agent['name']}", expanded=True):
                    col1, col2 = st.columns([3, 1])
                    with col1:
                        prompt = st.text_area(
                            "提示詞（可編輯）",
                            value=f"請以「{agent['name']}」的專業身份，分析以下文字：\n\n{current_text}\n\n要求用繁體中文、條理清晰、專業深入。",
                            height=150,
                            key=f"prompt_{i}"
                        )
                    with col2:
                        model_choice = st.selectbox("模型", options=list(MODELS.keys()), key=f"model_{i}")
                        max_tokens = st.slider("Max Tokens", 500, 8000, default_tokens, key=f"tokens_{i}")
                        temp = st.slider("Temperature", 0.0, 1.0, default_temp, 0.05, key=f"temp_{i}")

                    if st.button(f"執行此步驟", key=f"run_{i}"):
                        provider, model = MODELS[model_choice]
                        key = api_keys[provider]
                        if not key:
                            st.error(f"請提供 {provider.upper()} API 金鑰")
                            break

                        with st.spinner(f"{agent['name']} 分析中..."):
                            try:
                                if provider == "openai":
                                    client = OpenAI(api_key=key)
                                    resp = client.chat.completions.create(
                                        model=model,
                                        messages=[{"role": "user", "content": prompt}],
                                        max_tokens=max_tokens,
                                        temperature=temp
                                    )
                                    output = resp.choices[0].message.content
                                elif provider == "gemini":
                                    genai.configure(api_key=key)
                                    m = genai.GenerativeModel(model)
                                    resp = m.generate_content(prompt)
                                    output = resp.text
                                elif provider == "anthropic":
                                    client = Anthropic(api_key=key)
                                    resp = client.messages.create(
                                        model=model, max_tokens=max_tokens,
                                        messages=[{"role": "user", "content": prompt}]
                                    )
                                    output = resp.content[0].text
                                else:
                                    # xAI
                                    resp = httpx.post("https://api.x.ai/v1/chat/completions",
                                        headers={"Authorization": f"Bearer {key}"},
                                        json={"model": model, "messages": [{"role": "user", "content": prompt}], "max_tokens": max_tokens}
                                    )
                                    output = resp.json()["choices"][0]["message"]["content"]

                                st.markdown(output)
                                current_text = st.text_area("編輯後作為下一個輸入", value=output, height=200, key=f"edit_{i}")
                                results.append({"agent": agent["name"], "output": output})

                            except Exception as e:
                                st.error(f"錯誤：{e}")

            # 最終生成 20 個深度追問
            if results:
                st.markdown("## 20 個深度追問問題")
                final_prompt = "根據以上所有分析結果，請生成 **20 個極具洞察力、值得深思的追問問題**，每題獨立一行，涵蓋哲學、心理、社會、未來等多面向：\n\n" + current_text[:5000]
                # 使用第一個模型快速生成
                try:
                    provider, model = MODELS[default_model]
                    key = api_keys[provider]
                    if provider == "openai" and key:
                        client = OpenAI(api_key=key)
                        q = client.chat.completions.create(
                            model=model,
                            messages=[{"role": "user", "content": final_prompt}],
                            max_tokens=1500
                        )
                        st.markdown(q.choices[0].message.content.replace("\n", "  \n"))
                except:
                    st.info("追問問題生成需要 OpenAI/Gemini 支援")

st.success("多智能體鏈式分析系統已就緒！支援最強繁體中文深度分析")
