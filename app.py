import os
import tempfile
import subprocess
from flask import Flask, request, jsonify, render_template, Response
from dotenv import load_dotenv
import dashscope
from dashscope.audio.asr import Recognition
from openai import OpenAI

load_dotenv()

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 100 * 1024 * 1024  # 100MB

DASHSCOPE_API_KEY = os.getenv('DASHSCOPE_API_KEY')
DEEPSEEK_API_KEY = os.getenv('DEEPSEEK_API_KEY')

REVIEW_PROMPT = """你是一个顶尖的AI产品经理面试官和职业导师。我会给你一段面试录音的文字转录。

请按以下结构输出复盘报告，使用适合飞书文档的 Markdown 格式。

**重要格式要求：**
1. 每个段落之间必须空一行
2. 每个小节之间必须空两行
3. 列表项之间不空行，但列表前后要空行

---

# 面试复盘报告

## 1. 面试通过概率评估

[空一行]

客观评估候选人面试通过的概率（0-100%），并说明主要理由。每个理由单独成段，段落之间空一行。

[空两行]

## 2. 逐题深度点评

[空一行]

### 问题 1：[问题标题]

[空一行]

**候选人回答：**

[空一行]

> [引用原始回答，如果较长可以分段，段落之间空一行]

[空一行]

**点评：**

[空一行]

- 优点：[具体优点]
- 不足：[具体不足]

[空一行]

**Overqualified 参考答案：**

[空一行]

[顶尖候选人会怎么回答，给出具体示例。如果内容较长，分成多个段落，段落之间空一行]

[空两行]

### 问题 2：[问题标题]

[继续按照问题1的格式...]

[空两行]

## 3. 针对性学习建议

[空一行]

根据面试表现，给出3-5条犀利、具体、可执行的学习建议。不要泛泛而谈，要直指要害。

[空一行]

1. [建议1 - 详细描述，如果较长可以分成多句]

2. [建议2 - 详细描述]

3. [建议3 - 详细描述]

[空一行]

---

**输出时请严格遵守空行规则，确保段落之间有明显的视觉分隔。**

---
以下是面试转录文字：

{transcript}
"""


def convert_to_mono(input_path):
    """用 ffmpeg 转换为单声道 16kHz wav"""
    output_path = input_path + '_mono.wav'
    subprocess.run(
        ['/opt/homebrew/bin/ffmpeg', '-i', input_path, '-ac', '1', '-ar', '16000', '-y', output_path],
        check=True, capture_output=True
    )
    return output_path


def transcribe_audio(file_path):
    mono_path = convert_to_mono(file_path)
    try:
        dashscope.api_key = DASHSCOPE_API_KEY
        recognition = Recognition(
            model='paraformer-realtime-v2',
            format='wav',
            sample_rate=16000,
            callback=None
        )
        result = recognition.call(mono_path)
        if result.status_code == 200:
            sentences = result.get_sentence()
            transcript = ' '.join([s['text'] for s in sentences if s.get('text')])
            return transcript
        else:
            raise Exception('语音识别失败: ' + str(result.message))
    finally:
        os.unlink(mono_path)


def analyze_interview(transcript):
    client = OpenAI(
        api_key=DEEPSEEK_API_KEY,
        base_url='https://api.deepseek.com'
    )
    response = client.chat.completions.create(
        model='deepseek-chat',
        messages=[{'role': 'user', 'content': REVIEW_PROMPT.format(transcript=transcript)}],
        stream=True
    )
    for chunk in response:
        if chunk.choices[0].delta.content:
            yield chunk.choices[0].delta.content


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/transcribe', methods=['POST'])
def transcribe():
    if 'audio' not in request.files:
        return jsonify({'error': '没有上传文件'}), 400

    audio_file = request.files['audio']
    suffix = '.' + audio_file.filename.rsplit('.', 1)[-1].lower() if '.' in audio_file.filename else '.wav'

    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        audio_file.save(tmp.name)
        tmp_path = tmp.name

    try:
        transcript = transcribe_audio(tmp_path)
        return jsonify({'transcript': transcript})
    except Exception as e:
        return jsonify({'error': str(e)}), 500
    finally:
        os.unlink(tmp_path)


@app.route('/analyze', methods=['POST'])
def analyze():
    data = request.get_json()
    transcript = data.get('transcript', '').strip()
    if not transcript:
        return jsonify({'error': '转录文字为空'}), 400

    def generate():
        try:
            for chunk in analyze_interview(transcript):
                yield 'data: ' + chunk + '\n\n'
            yield 'data: [DONE]\n\n'
        except Exception as e:
            yield 'data: [ERROR] ' + str(e) + '\n\n'

    return Response(generate(), mimetype='text/event-stream',
                    headers={'Cache-Control': 'no-cache', 'X-Accel-Buffering': 'no'})


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5001))
    app.run(host='0.0.0.0', port=port, debug=False)
