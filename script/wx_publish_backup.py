# script/wx_publish_backup.py

# wx_publish_backup.py  (year-only folders; HTML named as YYYY-MM-DD_<title>.html)
import os, re, json, time, pathlib, hashlib, html, random
import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse, parse_qs
import mimetypes

# ===== 从env.json读取配置 =====
def load_env_config():
    """从env.json文件读取配置"""
    try:
        # 获取脚本所在目录的上级目录（项目根目录）
        script_dir = pathlib.Path(__file__).parent
        project_root = script_dir.parent
        env_file = project_root / "env.json"
        
        with open(env_file, "r", encoding="utf-8") as f:
            config = json.load(f)
        return config
    except FileNotFoundError:
        raise FileNotFoundError("请创建env.json文件，参考env.json.EXAMPLE")
    except json.JSONDecodeError:
        raise ValueError("env.json文件格式错误")

# 加载配置
config = load_env_config()
COOKIE = config.get("COOKIE", "")
TOKEN = config.get("TOKEN", "")  # 发表记录页URL里的 token
WECHAT_ACCOUNT_NAME = config.get("WECHAT_ACCOUNT_NAME", "unknown")

# ===== 可调参数 =====
COUNT_PER_PAGE = int(config.get("COUNT", "10"))   # 建议 10（接口上限）
SLEEP_LIST = float(config.get("SLEEP_LIST", "25"))  # 列表页间隔(秒) 2.5 起 # 我加得比较高
SLEEP_ART  = float(config.get("SLEEP_ART",  "15"))  # 单篇抓取间隔(秒) 1.5 起 # 我也往高了加
IMG_SLEEP  = float(config.get("IMG_SLEEP",  "0.8")) # 单张图片间隔(秒) 0.08起 # 也是扩大了十倍

# 新增：时间窗（env.json 可选填 START_DATE / END_DATE，格式 YYYY-MM-DD）
START_DATE = config.get("START_DATE", "")  # 例如 "2015-01-01"
END_DATE   = config.get("END_DATE", "")    # 例如 "2035-12-31"

def to_ts(dstr: str, end=False) -> int | None:
    if not dstr:
        return None
    t = time.strptime(dstr, "%Y-%m-%d")           # 以本地时区解析
    ts = int(time.mktime(t))                      # 当天 00:00:00
    return ts + (24*3600 - 1) if end else ts

START_TS = to_ts(START_DATE, end=False)
END_TS   = to_ts(END_DATE,   end=True)


script_dir = pathlib.Path(__file__).resolve().parent
project_root = script_dir.parent
OUTDIR = project_root / "Wechat-Backup" / WECHAT_ACCOUNT_NAME

TIMEOUT = 100

BASE = "https://mp.weixin.qq.com"
LIST_ENDPOINT = f"{BASE}/cgi-bin/appmsgpublish"

S = requests.Session()
S.headers.update({
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/139.0.0.0 Safari/537.36",
    "Cookie": COOKIE,
    "Referer": "https://mp.weixin.qq.com/",
    "Accept": "application/json,text/html;q=0.9,*/*;q=0.8",
    "Accept-Language": "zh-CN,zh;q=0.9",
    "X-Requested-With": "XMLHttpRequest"
})

from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

retry = Retry(
    total=5,
    backoff_factor=0.3,
    status_forcelist=[429, 500, 502, 503, 504],
    allowed_methods={"GET"},
)
S.mount("https://", HTTPAdapter(max_retries=retry))
S.mount("http://",  HTTPAdapter(max_retries=retry))


OUTDIR.mkdir(parents=True, exist_ok=True)
STATE_FILE = OUTDIR / "_state.json"
seen = set(json.loads(STATE_FILE.read_text("utf-8")).get("seen", [])) if STATE_FILE.exists() else set()

def save_state():
    STATE_FILE.write_text(json.dumps({"seen": list(seen)}, ensure_ascii=False, indent=2), "utf-8")

def sleep_with_jitter(base: float, jitter_ratio: float = 0.2):
    # 在 [base*(1-0.1), base*(1+0.1)] 之间随机
    jitter = base * jitter_ratio * (random.random() - 0.5)
    time.sleep(max(0.0, base + jitter))


def sanitize(name: str) -> str:
    s = re.sub(r"[\\/:*?\"<>|\r\n]+", "_", (name or "untitled")).strip()
    return s[:120] if len(s) > 120 else s

def fetch_publish_page(begin: int, count: int) -> dict:
    """优先走 JSON；失败则从HTML里的 'publish_page = {...};' 抽取"""
    params = {
        "sub": "list",
        "begin": begin,
        "count": count,
        "token": TOKEN,
        "lang": "zh_CN",
        "f": "json",
        "ajax": "1",
    }
    
    # a maybe key need
    params.update({"sub_action": "list_ex", "free_publish_type": "1"})

    r = S.get(LIST_ENDPOINT, params=params, timeout=TIMEOUT)
    r.raise_for_status()
    # 1) JSON
    try:
        data = r.json()
        pp = data.get("publish_page")
        if isinstance(pp, str):
            return json.loads(pp)
        elif isinstance(pp, dict):
            return pp
    except Exception:
        pass
    # 2) HTML 变量
    html_text = r.text
    m = re.search(r"publish_page\s*=\s*(\{[\s\S]*?\});", html_text)
    if not m:
        raise RuntimeError("未找到 publish_page")
    publish_page_str = html.unescape(m.group(1))
    return json.loads(publish_page_str)

def html_to_md(fragment: str) -> str:
    soup = BeautifulSoup(fragment, "html.parser")

    # 1) 先把 <a> 标签替换为 Markdown 语法，避免后续 get_text 丢失 href
    for a in soup.find_all("a"):
        href = a.get("href") or a.get("data-link") or ""
        text = a.get_text(" ", strip=True)
        if href:
            a.replace_with(f"[{text}]({href})")
        else:
            a.replace_with(text)

    lines = []
    for node in soup.descendants:
        if not getattr(node, "name", None):
            continue
        tag = node.name.lower()
        if tag in ("h1", "h2", "h3"):
            level = {"h1":"#","h2":"##","h3":"###"}[tag]
            lines.append(f"{level} {node.get_text(' ', strip=True)}\n")
        elif tag == "p":
            txt = node.get_text(" ", strip=True)
            if txt:
                lines.append(txt + "\n")
        elif tag == "img" and node.get("src"):
            alt = node.get("alt") or ""
            lines.append(f"![{alt}]({node.get('src')})\n")
        elif tag == "blockquote":
            lines.append("> " + node.get_text(" ", strip=True) + "\n")
        elif tag == "li":
            lines.append("- " + node.get_text(" ", strip=True))
        elif tag == "pre":
            lines.append("```\n" + node.get_text("", strip=True) + "\n```")
    return "\n".join(lines)


def wrap_full_html(body_html: str, title: str) -> str:
    # 轻量页面骨架，确保本地 file:// 直接可渲染
    return f"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width,initial-scale=1">
  <title>{title}</title>
  <style>
    body{{margin:16px; line-height:1.7; font-family:-apple-system,BlinkMacSystemFont,Segoe UI,Roboto,Helvetica,Arial,\\5FAE\\8F6F\\96C5\\9ED1,sans-serif}}
    img{{max-width:100%; height:auto}}
    pre,code{{font-family:ui-monospace,SFMono-Regular,Menlo,Consolas,monospace}}
    blockquote{{border-left:4px solid #ddd; padding-left:12px; color:#555}}
    /* 关键：离线时强制把微信正文容器显示出来 */
    #js_content, .rich_media_content{{visibility:visible !important; opacity:1 !important}}
  </style>
</head>
<body>
{body_html}
</body>
</html>"""


def download(url: str) -> str:
    r = S.get(url, timeout=TIMEOUT)
    r.raise_for_status()
    return r.text

def localize_images(content, folder: pathlib.Path, referer_link: str = ""):
    imgdir = folder / "images"
    imgdir.mkdir(exist_ok=True)
    soup = BeautifulSoup(content, "html.parser")
    idx = 1
    for img in soup.find_all("img"):
        src = img.get("data-src") or img.get("src")
        if not src or src.startswith("data:"):
            continue
        full = urljoin(BASE, src)
        try:
            headers = {"Referer": referer_link} if referer_link else {}
            resp = S.get(full, headers=headers, timeout=TIMEOUT)
            if not resp.ok or not resp.content:
                continue
            ext = ".jpg"
            ctype = (resp.headers.get("Content-Type") or "").split(";")[0].strip().lower()
            guess = mimetypes.guess_extension(ctype) if ctype else None
            if guess:
                ext = ".jpg" if guess == ".jpe" else guess
            m = re.search(r"wx_fmt=(\w+)", full)
            if m: ext = "." + m.group(1)
            fn = f"img_{idx:03d}{ext}"
            (imgdir / fn).write_bytes(resp.content)
            img["src"] = f"./images/{fn}"
            if "data-src" in img.attrs: del img["data-src"]
            idx += 1
            sleep_with_jitter(IMG_SLEEP, 1.2)
        except Exception:
            continue
    return str(soup)


def date_str_from_ts(ts: int) -> tuple[str, str]:
    """返回 (YYYY, YYYY-MM-DD)；微信时间戳为 UTC+8，本地用 time.localtime 即可"""
    t = time.localtime(ts or int(time.time()))
    return time.strftime("%Y", t), time.strftime("%Y-%m-%d", t)

# 新增：稳定的去重键
def key_from_link(link: str) -> str:
    try:
        u = urlparse(link)
        q = parse_qs(u.query)
        mid = (q.get("mid") or [None])[0]
        idx = (q.get("idx") or [None])[0]
        if mid and idx:
            return f"{mid}_{idx}"
    except Exception:
        pass
    return hashlib.md5(link.encode("utf-8")).hexdigest()

def save_article(title: str, link: str, ts: int):
    # 去重
    key = key_from_link(link)
    if key in seen:
        return

    html_text = download(link)
    soup = BeautifulSoup(html_text, "html.parser")
    content = soup.select_one("#js_content") or soup.body or soup

    # 去掉 js_content 上的隐藏样式，避免离线看不到
    try:
        st = content.get("style","")
        st = re.sub(r"(?:^|;)\s*(?:visibility\s*:\s*hidden|opacity\s*:\s*0)\s*;?", "", st, flags=re.I)
        if st.strip(): content["style"] = st
        else: content.attrs.pop("style", None)
    except Exception:
        pass

    year, ymd = date_str_from_ts(ts)
    safe_title = sanitize(title)
    base_name = f"{ymd}_{safe_title}"              # 用于文件命名
    year_dir = OUTDIR / year                       # 只按年份分目录
    article_dir = year_dir / base_name             # 仍保留每篇一个目录（用于 images）
    article_dir.mkdir(parents=True, exist_ok=True)

    # 图片本地化
    localized = localize_images(str(content), article_dir, link)

    # 生成完整 HTML（可直接双击打开）
    full_html = wrap_full_html(localized, safe_title)

    # 文件名要求：YYYY-MM-DD_<标题>.html / .md
    (article_dir / f"{base_name}.html").write_text(full_html, "utf-8")
    (article_dir / f"{base_name}.md").write_text(html_to_md(localized), "utf-8")

    meta = {"title": title, "url": link, "publish_time": ts, "year": year, "date": ymd}
    (article_dir / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), "utf-8")

    seen.add(key); save_state()
    print("saved:", (article_dir / f"{base_name}.html").relative_to(OUTDIR))

def extract_links_from_publish_item(item: dict):
    """单次群发里的子文章列表"""
    links = []
    info = item.get("publish_info")
    if not info:
        return links
    if isinstance(info, str):
        info = html.unescape(info)
        try:
            info = json.loads(info)
        except Exception:
            return links
    for a in info.get("appmsgex", []):
        title = a.get("title") or a.get("appmsg_title") or ""
        link  = a.get("link") or a.get("content_url") or a.get("url") or ""
        ts    = a.get("create_time") or a.get("comm_msg_info", {}).get("datetime") or 0
        if link:
            link = urljoin(BASE, link)
            links.append((title, link, ts))
    return links


# 预检
def probe() -> bool:
    """探针请求：确认 COOKIE/TOKEN 有效且能拿到 publish_page"""
    print("→ 正在预检 appmsgpublish 接口…")
    try:
        page = fetch_publish_page(0, 1)  # begin=0,count=1 最小负载
    except requests.HTTPError as e:
        code = getattr(e.response, "status_code", None)
        print(f"❌ 预检失败：HTTP {code}，请刷新后台后重新复制 Cookie/Token")
        return False
    except Exception as e:
        print("❌ 预检异常：", repr(e))
        return False

    # fetch_publish_page() 会返回已解析的 publish_page 对象（dict）
    if not isinstance(page, dict):
        print("⚠️ 预检返回非 dict，可能被重定向到登录页；请更新 Cookie/Token")
        return False

    # 常见结构：{"publish_list":[...], "total_count": 433, ...}
    publish_list = page.get("publish_list")
    total = page.get("total_count")
    if isinstance(publish_list, list):
        print(f"✅ 预检通过：publish_list={len(publish_list)}，total_count={total}")
        return True

    # 某些异常场景：返回 {} 或者缺少关键字段
    print("⚠️ 预检未见 publish_list，疑似登录态失效或权限不足；请在 DevTools 的 Network 复制最新的 Request Headers → Cookie")
    return False


def main():
    begin = 0
    page_idx = 0
    total = None

    while True:
        # 先按配置的 COUNT_PER_PAGE 拉
        page = fetch_publish_page(begin, COUNT_PER_PAGE)
        publish_list = page.get("publish_list") or []
        total = total if total is not None else page.get("total_count", None)

        # 如果空，降级重试：count=1（很多号只在 count=1 时返回）
        if not publish_list and COUNT_PER_PAGE != 1:
            page = fetch_publish_page(begin, 1)
            publish_list = page.get("publish_list") or []

        print(f"[list] begin={begin} count_req={COUNT_PER_PAGE} -> got={len(publish_list)}"
              + (f" total={total}" if total is not None else ""))

        if not publish_list:
            print("→ 列表为空，结束。若非预期，请降低 COUNT 到 1~10 再试。")
            break

        # 逐条保存
        for item in publish_list:
            for title, link, ts in extract_links_from_publish_item(item):
                # 时间窗过滤（默认不过滤）
                if (START_TS and ts and ts < START_TS) or (END_TS and ts and ts > END_TS):
                    continue
                save_article(title, link, ts)
                sleep_with_jitter(SLEEP_ART)

        # 用“实际返回条数”推进 begin，避免跳页或空页
        begin += len(publish_list)
        page_idx += 1
        sleep_with_jitter(SLEEP_LIST)

# --- program entry ---
if __name__ == "__main__":
    # 1) 基本配置检查
    if not (COOKIE and TOKEN):
        raise SystemExit("请在 env.json 中配置 COOKIE 和 TOKEN")
    OUTDIR.mkdir(parents=True, exist_ok=True)

    # 2) 先跑探针（只请求一页，确认 publish_page / publish_list 可见）
    ok = probe()
    if not ok:
        raise SystemExit("预检未通过：Cookie/Token 可能失效，请在 DevTools 的 Request Headers 复制最新 Cookie")

    # 3) 通过后再正式抓取
    main()

    print("✅ 完成 →", OUTDIR.resolve())


