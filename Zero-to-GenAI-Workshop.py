# Databricks notebook source
# MAGIC %md
# MAGIC
# MAGIC %md
# MAGIC <img src="https://github.com/dbrxkr/zero-to-gen-ai/blob/main/img/zero-to-genai-header-black.png?raw=true" style="float: right; width: 100%; margin-left: 10px">
# MAGIC
# MAGIC

# COMMAND ----------

# MAGIC %md-sandbox
# MAGIC
# MAGIC # 1. LLM 챗봇 RAG를 위한 준비
# MAGIC
# MAGIC ## Databricks Vector Search를 활용하여 지식기반 RAG 시스템 구축하기
# MAGIC
# MAGIC 이 노트북은 Zero to GenAI 워크샵을 위해서 Databricks 및 AWS의 LLM 기능들을 활용하여 RAG 챗봇을 개발합니다.
# MAGIC
# MAGIC 개발되는 RAG 챗봇은 파운데이션에 없는 정보를 답변할 수 있도록 PDF 문서를 작은 덩어리(chunk)로 구분하고 Vector Search에 인덱싱 합니다.
# MAGIC
# MAGIC RAG 시스템의 가장 중요한 요소는 고품질의 데이터를 준비하는 것입니다. 본 워크샵 이후에 직접 사내 데이터를 사용하여 RAG 시스템을 구축하는 경우 고품질의 데이터를 준비할 필요가 있습니다.
# MAGIC
# MAGIC
# MAGIC 본 워크샵의 예제에서는 대한민국 과학기술정보통신부 산하의 소프트웨어정책연구소의 연구자료 문서를 사용하겠습니다(https://spri.kr/). 
# MAGIC - PDF문서는 실습용 zip 파일을 '가져오기' 할때 포함되어 있습니다. 
# MAGIC - 페이지를 작은 텍스트 덩어리(청크)로 분할합니다.
# MAGIC - 델타 테이블의 일부로 데이터브릭스 임베딩 모델을 사용하여 임베딩 벡터값을 계산합니다.
# MAGIC - 델타 테이블을 기반으로 Databricks Vector Search 인덱스를 생성합니다.

# COMMAND ----------

# MAGIC %md
# MAGIC ## 워크샵에서 개발할 RAG 시스템의 구조도

# COMMAND ----------

# MAGIC %md
# MAGIC <img src="https://github.com/dbrxkr/zero-to-gen-ai/blob/main/img/zero-to-genai-image01.png?raw=true" style="float: right; width: 100%; margin-left: 10px">

# COMMAND ----------

# MAGIC %md
# MAGIC ## Databricks on AWS에서 RAG 실습을 위한 환경설정을 진행 합니다
# MAGIC - 관련된 Python 패키지를 설치합니다.
# MAGIC - 실습에 도움이되는 Helper 함수들을 정의합니다.

# COMMAND ----------

# DBTITLE 1,RAG 구현이 필요로 되는 Python 패키지 설치
!pip3 install -qqqq --upgrade pip
!pip3 install -qqqq mlflow==2.10.1 lxml==4.9.3 transformers==4.34.0 langchain==0.1.20 beautifulsoup4==4.12.2 pymupdf4llm==0.0.10 aiohttp==3.10.0 gradio==4.0.0 #3.50.2
!pip3 install -qqqq dbtunnel[gradio] databricks-vectorsearch==0.22 databricks-sdk databricks databricks-genai-inference
dbutils.library.restartPython()

# COMMAND ----------

# MAGIC %md
# MAGIC ## 👇 사용자의 개별 환경에 따라 아래 정보를 업데이트 해주세요. 👇

# COMMAND ----------

# DBTITLE 1,개별 환경 변수 설정
# 환경 변수 설정. 실제 운영 환경에서는 Databricks Secrets 환경 변수로 설정 하여 보안을 강화해야합니다.
uc_catalog = catalog = "<자신의 카탈로그명>" # 자신의 카탈로그명 : catalog_<12자리_AWS_Account_ID>
databricks_token = "<Databricks 토큰>" # Databricks PAT에서 발급받은 dapi로 시작하는 토큰
aws_access_key_id = "<자신의 AWS AccessKey>" # 자신의 AWS AccessKey 입력
aws_secret_access_key = "<자신의 AWS SecretAccessKey>" # 자신의 AWS SecretAccessKey 입력

# COMMAND ----------

# DBTITLE 1,공통 변수 설정
bedrock_region = "us-west-2" # 교육용 리전인 us-west-2 입력
uc_schema = schema = db = dbName = "schema_rag" # 그대로 사용
embedding_model_name = "embedding_model" # 그대로 사용
generative_model_name = "foundation_model" # 그대로 사용

host = "https://" + spark.conf.get("spark.databricks.workspaceUrl")
print(host)

# COMMAND ----------

# DBTITLE 1,Helper 함수 초기화
# MAGIC %run ./init-script $reset_all_data=true

# COMMAND ----------

# MAGIC %md-sandbox
# MAGIC
# MAGIC ## Amazon Bedrock 연결을 위한 AccessKey와 SecretAccessKey 등록
# MAGIC <br/>
# MAGIC <div style="background-color: #def2ff; padding: 15px; boarder: 30px; ">
# MAGIC   <strong>✅ 정 보</strong><br/>
# MAGIC   - 아래 작업은 민감정보인 AWS AccessKey와 SecretAccessKey를 Databricks Secrets 저장소에 저장하는 절차입니다. <br/>
# MAGIC   - 본 교육에서는 교육의 목적상, AccessKey, SecretAccessKey, PersonalAccessKey 등을 본 노트북에 노출하여 작업하지만, 실제 운영 환경에서는 클라이언트 환경에서 Databricks CLI를 설치하여 관리자가 Databricks Secrets 저장소를 사용하여 노출되지 않게 관리 및 사용되어야 합니다.
# MAGIC </div>

# COMMAND ----------

# DBTITLE 1,Databricks Secret 저장소에 secret scope 생성
import requests
import json

url = f"{host}/api/2.0/secrets/scopes/create"

payload = json.dumps({
  "scope": "zero-to-scope",
  "scope_backend_type": "DATABRICKS"
})
headers = {
  'Content-Type': 'application/json',
  'Authorization': f'Bearer {databricks_token}'
}

response = requests.request("POST", url, headers=headers, data=payload)

print(response.text)


# COMMAND ----------

# DBTITLE 1,Databricks Secret 저장소에 AWS access key값 저장
import requests
import json

url = f"{host}/api/2.0/secrets/put"

payload = json.dumps({
  "scope": "zero-to-scope",
  "key": "access_key",
  "string_value": f"{aws_access_key_id}"
})
headers = {
  'Content-Type': 'application/json',
  'Authorization': f'Bearer {databricks_token}'
}

response = requests.request("POST", url, headers=headers, data=payload)

print(response.status_code)


# COMMAND ----------

# DBTITLE 1,Databricks Secret 저장소에 AWS access access key값 저장
import requests
import json

url = f"{host}/api/2.0/secrets/put"

payload = json.dumps({
  "scope": "zero-to-scope",
  "key": "secret_access_key",
  "string_value": f"{aws_secret_access_key}"
})
headers = {
  'Content-Type': 'application/json',
  'Authorization': f'Bearer {databricks_token}'
}

response = requests.request("POST", url, headers=headers, data=payload)

print(response.status_code)


# COMMAND ----------

# DBTITLE 1,Databricks Secret 저장소에 저장된 key들 확인
import requests
import json

url = f"{host}/api/2.0/secrets/list?scope=zero-to-scope"

payload = json.dumps({
  "scope": "zero-to-scope",
  "scope_backend_type": "DATABRICKS"
})
headers = {
  'Content-Type': 'application/json',
  'Authorization': f'Bearer {databricks_token}'
}

response = requests.request("GET", url, headers=headers, data=payload)

print(response.text)


# COMMAND ----------

# MAGIC %md
# MAGIC ## 한국어 임베딩 및 생성을 위한 Amazon Bedrock LLM model 엔드포인트 등록
# MAGIC
# MAGIC 본 RAG 실습에서는 2개의 대규모 언어 모델(LLM)을 사용할 것입니다.
# MAGIC
# MAGIC - 임베딩 모델(Embedding Model) : Databricks Vertor Search에 저장 및 검색에 사용할 문장 임베딩을 위한 언어 모델
# MAGIC - 문장 생성 파운데이션 모델(Genaration Foundation Model) : Databricks Vector Search의 검색 결과를 바탕으로 사용자에게 답변을 생성하기 위한 언어 모델
# MAGIC
# MAGIC Databricks는 내재된 다양한 임베딩 모델과 파운데이션 모델을 제공하지만, 본 워크샵에서는 Aamazon Bedrock을 연계하여 RAG 시스템을 구축할 것입니다. 
# MAGIC
# MAGIC - 임베딩 모델 : Amazon Bedrock Titan embed g1 모델
# MAGIC - 문장 생성 파운데이션 모델 : Amazon Bedrock Claude 3 sonnet 모델
# MAGIC

# COMMAND ----------

# MAGIC %md
# MAGIC <img src="https://github.com/dbrxkr/zero-to-gen-ai/blob/main/img/zero-to-genai-image00.png?raw=true" style="float: right; width: 100%; margin-left: 10px">

# COMMAND ----------

# DBTITLE 1,임베딩 모델 엔드포인트 생성
import mlflow.deployments
deploy_client = mlflow.deployments.get_deploy_client("databricks")

if embedding_model_name in [endpoints['name'] for endpoints in deploy_client.list_endpoints()]:    
    #deploy_client.delete_endpoint(embedding_model_name)
    print("동일한 모델의 엔드포인트가 이미 존재합니다.")

if embedding_model_name not in [endpoints['name'] for endpoints in deploy_client.list_endpoints()]:    
    deploy_client.create_endpoint(
        name=embedding_model_name,
        config={
            "served_entities": [
                {
                    "external_model": {
                        "name": "titan-embed-g1-text-02",
                        "provider": "amazon-bedrock",
                        "task": "llm/v1/embeddings",
                        "amazon_bedrock_config": {
                            "aws_region": f"{bedrock_region}",
                            "aws_access_key_id": "{{secrets/zero-to-scope/access_key}}",
                            "aws_secret_access_key": "{{secrets/zero-to-scope/secret_access_key}}",
                            "bedrock_provider": "amazon",
                        },
                    }
                }
            ]
        },
    )
    print("엔드포인트가 배포 되었습니다.")


# COMMAND ----------

# DBTITLE 1,생성 파운데이션 모델 엔드포인트 생성
import mlflow.deployments
deploy_client = mlflow.deployments.get_deploy_client("databricks")

if generative_model_name in [endpoints['name'] for endpoints in deploy_client.list_endpoints()]:    
    #deploy_client.delete_endpoint(generative_model_name)
    print("동일한 모델의 엔드포인트가 이미 존재합니다.")

# claude-3-5-sonnet-20240620-v1:0, claude-3-sonnet-20240229-v1:0
if generative_model_name not in [endpoints['name'] for endpoints in deploy_client.list_endpoints()]:    
    deploy_client.create_endpoint(
        name=generative_model_name,
        config={
            "served_entities": [
                {
                    "external_model": {
                        "name": "claude-3-sonnet-20240229-v1:0",
                        "provider": "amazon-bedrock",
                        "task": "llm/v1/chat",
                        "amazon_bedrock_config": {
                            "aws_region": f"{bedrock_region}",
                            "aws_access_key_id": "{{secrets/zero-to-scope/access_key}}",
                            "aws_secret_access_key": "{{secrets/zero-to-scope/secret_access_key}}",
                            "bedrock_provider": "anthropic",
                        },
                    }
                }
            ]
        },
    )
    print("엔드포인트가 배포 되었습니다.")


# COMMAND ----------

# DBTITLE 1,위 엔드포인트 배포의 안정성을 위해 약간의 시간을 대기
# 모델 배포가 완료되기까지 5분 정도 기다립니다.
import time
time.sleep(300)

# COMMAND ----------

# MAGIC %md-sandbox
# MAGIC # 2. PDF문서에서 문서를 Markdown 형태로 추출하기
# MAGIC
# MAGIC 저장한 PDF를 Databricks Vector Search에 저장하기위해 처리해야 하는 단계는 다음과 같습니다:
# MAGIC
# MAGIC - PDF 문서를 로드하여 추출 페이지, 추출 레이아웃을 지정합니다.
# MAGIC - pymupdf4llm 패키지를 사용하여 PDF문서의 내용을 Markdown 포멧으로 추출합니다.
# MAGIC - 추출한 Markdown텍스트를 가공 합니다.
# MAGIC - 전처리된 Markdown텍스트를 분할합니다.
# MAGIC
# MAGIC PDF를 추출하기위해 PyMuPDF4LLM 패키지를 사용합니다.

# COMMAND ----------

# MAGIC %md
# MAGIC ## PyMuPDF4LLM 패키지 
# MAGIC - 이 패키지는 PyMuPDF를 사용하여 파일의 페이지를 마크다운 형식의 텍스트로 변환합니다.
# MAGIC - 표준 텍스트와 표를 감지하여 올바른 읽기 순서로 가져온 다음 GitHub 호환 마크다운 텍스트로 함께 변환합니다.
# MAGIC - 헤더 줄은 글꼴 크기를 통해 식별되며 하나 이상의 # 태그가 적절하게 접두사로 붙습니다.
# MAGIC - 굵게, 이탤릭체, 단일 간격 텍스트 및 코드 블록이 감지되고 그에 따라 형식이 지정됩니다. 정렬된 목록과 정렬되지 않은 목록에도 비슷하게 적용됩니다.
# MAGIC - 기본적으로 모든 문서 페이지가 처리됩니다. 원하는 경우 0을 기준으로 한 페이지 번호 목록을 제공하여 페이지의 하위 집합을 지정할 수 있습니다.
# MAGIC - 관련 매뉴얼 페이지(https://pymupdf4llm.readthedocs.io/en/latest/)
# MAGIC
# MAGIC 추출 파라미터:
# MAGIC - doc: 문서 또는 문자열
# MAGIC - pages: 고려할 페이지 번호 목록(0부터 시작).
# MAGIC - hdr_info: 'get_hdr_info'라는 메서드가 있는 콜러블 또는 객체
# MAGIC - write_images: (bool) 이미지/그림을 파일로 저장할지 여부
# MAGIC - page_chunks: (bool) 출력을 페이지별로 분할할지 여부
# MAGIC - margins: 콘텐츠가 겹치는 여백 영역을 고려하지 않습니다
# MAGIC - dpi: (int) 생성된 이미지의 원하는 해상도
# MAGIC - page_width: (float) 페이지 레이아웃이 가변적인 경우 가정
# MAGIC - page_height: 페이지 레이아웃이 가변인 경우 (float) 가정
# MAGIC - table_strategy: 테이블 감지 전략 선택
# MAGIC - graphics_limit: (int) 벡터 그래픽이 너무 많은 페이지 무시

# COMMAND ----------

# MAGIC %md
# MAGIC ## PDF 레이아웃내에 컨텐츠 영역을 지정하여 추출

# COMMAND ----------

# MAGIC %md
# MAGIC <img src="https://github.com/dbrxkr/zero-to-gen-ai/blob/main/img/zero-to-genai-data-prep01.png?raw=true" style="float: right; width: 100%; margin-left: 10px">

# COMMAND ----------

# DBTITLE 1,컨텐츠 영역을 지정하여 추출
import pymupdf4llm
from langchain.text_splitter import MarkdownTextSplitter

md_text = pymupdf4llm.to_markdown(doc="./krpdf.pdf"
                                  ,pages=list(range(5, 20))                              
                                  ,write_images=False
                                  ,margins=(20, 60, 20, 60) # 왼쪽, 위쪽, 오른쪽, 아래
                                  ,table_strategy='lines_strict'
                                  ,page_chunks=False
                                  #,graphics_limit=20
                                  ) 


print(md_text)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 레이아웃 손상으로 발생한 에러 메시지를 제거

# COMMAND ----------

# DBTITLE 1,눈에 보이지 않는 손상된 레이아웃으로 인해 발생한 에러 메시지 제거
import re

def remove_syntax_error_lines(text):
    # "syntax error"를 포함하는 모든 줄을 제거하는 패턴 정의
    pattern = re.compile(r'.*syntax error.*\n?')
    
    # 패턴에 맞는 줄을 빈 문자열로 대체
    cleaned_text = re.sub(pattern, '', text)
    
    return cleaned_text

# "syntax error"가 포함된 줄을 제거하는 함수 호출
md_text_temp = remove_syntax_error_lines(md_text)

print(md_text_temp)


# COMMAND ----------

# MAGIC %md
# MAGIC ## 불필요한 각주 제거

# COMMAND ----------

# MAGIC %md
# MAGIC <img src="https://github.com/dbrxkr/zero-to-gen-ai/blob/main/img/zero-to-genai-data-prep02.png?raw=true" style="float: right; width: 100%; margin-left: 10px">

# COMMAND ----------

# DBTITLE 1,모든 페이지의 각주를 제거
import re

def remove_footnotes_and_references(text):
    # 페이지 구분자 정의
    page_split_pattern = re.compile(r'-----')
    
    # 페이지별로 텍스트 분할
    pages = page_split_pattern.split(text)
    
    # 각 페이지에서 각주 내용을 제거하는 패턴 정의
    footnote_pattern = re.compile(r'\d+\) .*(\n.*)*')
    
    # 본문 내의 각주 참조 제거 패턴 정의
    reference_pattern = re.compile(r'\[\d+\)\]')
    
    cleaned_pages = []
    
    for page in pages:
        # 각주 내용을 제거
        cleaned_page = re.sub(footnote_pattern, '', page).strip()
        # 본문 내의 각주 참조 제거
        cleaned_page = re.sub(reference_pattern, '', cleaned_page).strip()
        cleaned_pages.append(cleaned_page)
    
    # 페이지 구분자를 다시 추가하여 결합
    cleaned_text = '\n\n-----\n'.join(cleaned_pages)
    
    return cleaned_text


# 각 페이지의 각주 내용을 제거하는 함수 호출
md_text_temp = remove_footnotes_and_references(md_text_temp)

print(md_text_temp)


# COMMAND ----------

# MAGIC %md
# MAGIC ## 문서에서 대/중/소분류 제목을 식별

# COMMAND ----------

# MAGIC %md
# MAGIC <img src="https://github.com/dbrxkr/zero-to-gen-ai/blob/main/img/zero-to-genai-data-prep03.png?raw=true" style="float: right; width: 100%; margin-left: 10px">

# COMMAND ----------

# DBTITLE 1,문서의 컨텐츠를 마크다운 대/중/소로 분류
import re

# ''을 ' '로 변경
md_text_temp = re.sub(r'', ' ', md_text_temp)

# '####'을 '# '로 변경
md_text_temp = re.sub(r'####', '#', md_text_temp)

# 숫자. 으로 시작하는 항목을 ## 숫자. 으로 변경
md_text_temp = re.sub(r'^(\d+)\. ', r'## \1. ', md_text_temp, flags=re.MULTILINE)

# '□'으로 시작하는 문장을 ### 으로 변경
md_text_temp = re.sub(r'^□', r'### ', md_text_temp, flags=re.MULTILINE)

print(md_text_temp)


# COMMAND ----------

# MAGIC %md
# MAGIC ## 한국 공공기관에서 자주 사용되는 기호를 제거 또는 치환

# COMMAND ----------

# DBTITLE 1,마크다운언어가 이해하지 못하는 특수기호 제거
import re

md_text_temp = re.sub(r'ㅇ', '- ', md_text_temp)
md_text_temp = re.sub(r'→', ' 에서 ', md_text_temp)
md_text_temp = re.sub(r'’', '20', md_text_temp)
md_text_temp = re.sub(r'☞', '- ', md_text_temp)
md_text_temp = re.sub(r'[\*\*\[\]▲「」]', '', md_text_temp)

print(md_text_temp)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 유의미한 정보를 담고 있는 괄호안의 내용은 남겨두고 인용 구문은 삭제

# COMMAND ----------

# MAGIC %md
# MAGIC <img src="https://github.com/dbrxkr/zero-to-gen-ai/blob/main/img/zero-to-genai-data-prep04.png?raw=true" style="float: right; width: 100%; margin-left: 10px">

# COMMAND ----------

# DBTITLE 1,모든 문서의 인용 구문 삭제
import re

# 괄호 안에 쉼표가 있는 경우 괄호와 그 안의 내용을 모두 삭제
md_text_temp = re.sub(r'\([^()]*,[^()]*\)', '', md_text_temp)

print(md_text_temp)


# COMMAND ----------

# MAGIC %md
# MAGIC ## 문서에서 표를 추출하여 마크다운 문서로 변환

# COMMAND ----------

# MAGIC %md
# MAGIC <img src="https://github.com/dbrxkr/zero-to-gen-ai/blob/main/img/zero-to-genai-data-prep05.png?raw=true" style="float: right; width: 100%; margin-left: 10px">

# COMMAND ----------

# DBTITLE 1,문서의 표를 마크다운으로 추출하고 레이아웃 정리
import re

# '-----' 기호를 ''로 치환
md_text_temp = re.sub(r'-----', '', md_text_temp)

# 모든 개행 삭제
md_text_temp = md_text_temp.replace('\n\n', '')

# 이중 공백 삭제
md_text_temp = md_text_temp.replace('  ', ' ')

# '#'가 등장하면 앞에 개행을 추가
md_text_temp = re.sub(r'(#+)', r'\n\n\1', md_text_temp)

# 'C'와 '#' 사이의 개행이나 공백을 제거
md_text_temp = re.sub(r'C\s*#', r'C#', md_text_temp)

# '-'가 등장하면 앞에 개행을 추가
# '-' 다음에 공백이 아닌 문자가 오는 경우에만 새로운 줄로 분리
# 즉, '-' 뒤에 공백이 오면 항목 구분자로 처리하지 않음
md_text_temp = re.sub(r'- (?=\S)', r'\n\n- ', md_text_temp.strip())

# '표' 앞에 두 번 개행 추가
md_text_temp = re.sub(r'(표 \d)', r'\n\n\1', md_text_temp)
# '표 2', '표 3', '표 4' 다음에 '|' 앞에서 두 번 개행 추가
md_text_temp = re.sub(r'(표 \d.*?)(\|)', r'\1\n\n\2', md_text_temp)

# '※'로 시작하는 문장과 그 후에 오는 관련 내용을 제거
md_text_temp = re.sub(r'※[^­]*­', '\n', md_text_temp, flags=re.DOTALL)

print(md_text_temp)


# COMMAND ----------

# MAGIC %md-sandbox
# MAGIC
# MAGIC # 3. 문서 페이지를 작은 덩어리(chunk)로 나누기
# MAGIC
# MAGIC LLM 모델에는 일반적으로 최대 입력 컨텍스트 길이가 있으며, 매우 긴 텍스트의 임베딩을 계산할 수 없습니다.
# MAGIC 또한 컨텍스트 길이가 길수록 모델이 응답을 제공하는 데 시간이 더 오래 걸립니다.
# MAGIC
# MAGIC 모델의 성능을 높이려면 문서 준비가 핵심이며, 데이터 세트에 따라 여러 가지 전략이 존재합니다:
# MAGIC
# MAGIC - 문서를 작은 덩어리(단락, 헤더)로 나누는 전략을 수행할 수 있습니다.(마크다운 또는 HTML로 식별)
# MAGIC - 문서를 고정된 길이(토큰 수)로 나누는 전략을 수행할 수 있습니다.
# MAGIC - 청크 크기는 콘텐츠와 이를 사용하여 프롬프트를 작성하는 방식에 따라 달라집니다. 프롬프트에 작은 문서 청크를 여러 개 추가하면 큰 청크만 보내는 것과 다른 결과가 나올 수 있습니다.
# MAGIC - 큰 청크로 나누고 모델에 각 청크를 일회성 작업으로 요약하도록 요청하면 실시간 추론 속도가 빨라집니다.
# MAGIC - 여러 에이전트를 만들어 각각의 큰 문서를 병렬로 평가하고 최종 에이전트에게 답변을 작성하도록 요청하는 전략을 구사할 수 있습니다.
# MAGIC

# COMMAND ----------

# MAGIC %md
# MAGIC <img src="https://github.com/dbrxkr/zero-to-gen-ai/blob/main/img/zero-to-genai-image02.png?raw=true" style="float: right; width: 100%; margin-left: 10px">

# COMMAND ----------

# MAGIC %md
# MAGIC ## MarkdownHeaderTextSplitter를 사용하여 PDF에서 추출한 내용을 Markdown형식으로 추출

# COMMAND ----------

# DBTITLE 1,MarkdownHeaderTextSplitter를 사용하여 청킹
from langchain_text_splitters import MarkdownHeaderTextSplitter

headers_to_split_on = [
                        ("#", "Header 1"),
                        ("##", "Header 2"),
                        ("###", "Header 3")
                      ]

# 텍스트를 마크다운 헤더를 기준으로 청크로 분할하는 MarkdownHeaderTextSplitter 객체를 생성
markdown_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers_to_split_on)

# 텍스트를 헤더를 기준으로 청크로 분할하여 리스트 객체로 저장
md_header_splits = markdown_splitter.split_text(md_text_temp)

# 청크를 출력하여 확인
for chunk in md_header_splits:
  print(f"{chunk.to_json}")
  print("-----------------------------------")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 청크정보를 임베딩하여 벡터값을 구한 뒤 델타 테이블에 저장

# COMMAND ----------

# DBTITLE 1,청크 정보를 저장할 델타 테이블을 생성
spark.sql(f"""
CREATE OR REPLACE TABLE {uc_catalog}.{uc_schema}.databricks_documentation (
  id BIGINT GENERATED BY DEFAULT AS IDENTITY
  ,created_at TIMESTAMP
  ,content STRING
) TBLPROPERTIES (delta.enableChangeDataFeed = true)
""").display()

# COMMAND ----------

# DBTITLE 1,테이블에 청크 정보를 저장
import json

for chunk in md_header_splits:
  json_chunk = json.dumps(chunk.to_json()).replace("'", "''")
  spark.sql(f"""
            INSERT INTO {uc_catalog}.{uc_schema}.databricks_documentation(content, created_at)
            VALUES('{json_chunk}', from_utc_timestamp(current_timestamp(), 'UTC+9'))
            """)

spark.sql(f"""
          SELECT * FROM {uc_catalog}.{uc_schema}.databricks_documentation ORDER BY id
          """).display()

# COMMAND ----------

# DBTITLE 1,Vector Store에 저장하기전 임베딩 모델 테스트
import time
time.sleep(60)

import mlflow.deployments
deploy_client = mlflow.deployments.get_deploy_client("databricks")

response = deploy_client.predict(endpoint=embedding_model_name, inputs={"input": ["Enlgish embedding Test."]})
embeddings = [e['embedding'] for e in response.data]
print(embeddings)

time.sleep(60)

response = deploy_client.predict(endpoint=embedding_model_name, inputs={"input": ["한국어 임베딩 테스트 입니다."]})
embeddings = [e['embedding'] for e in response.data]
print(embeddings)

time.sleep(60)

response = deploy_client.predict(endpoint=embedding_model_name, inputs={"input": ["Databricks와 AWS가 함께하는 Zero to GenAI."]})
embeddings = [e['embedding'] for e in response.data]
print(embeddings)

time.sleep(60)

# COMMAND ----------

# MAGIC %md
# MAGIC # 4. 델타 테이블에 저장한 데이터를 Databricks Vector Search에 동기화
# MAGIC
# MAGIC <img src="https://github.com/databricks-demos/dbdemos-resources/blob/main/images/product/chatbot-rag/databricks-vector-search-managed-type.png?raw=true" style="float: right" width="800px">
# MAGIC
# MAGIC Databricks의 Vector Search는 3가지 방식의 인덱스 검색을 제공합니다:
# MAGIC
# MAGIC - **관리형 임베딩**: 사용자가 텍스트 열과 엔드포인트 이름을 제공하면, 데이터브릭스에서 인덱스를 델타 테이블과 동기화합니다(본 실습에서 사용할 방법).
# MAGIC - **자체 관리 임베딩**: 임베딩을 직접 계산하여 델타 테이블의 필드로 저장하면, 데이터브릭스에서 인덱스를 동기화합니다.
# MAGIC - **직접 인덱스**: 델타 테이블 없이 임베딩을 직접 계산하여 Vector Search에 직접 입력 합니다.
# MAGIC
# MAGIC 이 실습에서는 **관리형 임베딩** 인덱스를 설정하는 방법을 보여드리겠습니다.
# MAGIC

# COMMAND ----------

# DBTITLE 1,Vector Search 엔드포인트 생성
from databricks.vector_search.client import VectorSearchClient
vsc = VectorSearchClient()

if not endpoint_exists(vsc, vector_search_endpoint_name):
    vsc.create_endpoint(name=vector_search_endpoint_name, endpoint_type="STANDARD")

print("Vector Search 엔드포인트를 배포합니다. 이 작업은 10~15분 정도 소요됩니다.")
wait_for_vs_endpoint_to_be_ready(vsc, vector_search_endpoint_name)
print(f"Vector Search 엔드포인트 {vector_search_endpoint_name} 가 준비 되었습니다.")

# COMMAND ----------

# DBTITLE 1,청크정보를 저장한 테이블을 임베딩한 뒤 Vector Search에 동기화
from databricks.sdk import WorkspaceClient
import databricks.sdk.service.catalog as c

# 색인하려는 테이블
source_table_fullname = f"{uc_catalog}.{uc_schema}.databricks_documentation"
# 인덱스를 저장할 위치
vs_index_fullname = f"{uc_catalog}.{uc_schema}.databricks_documentation_vs_index"

if not index_exists(vsc, vector_search_endpoint_name, vs_index_fullname):
  print(f"Vector Search 인덱스 {vs_index_fullname} 를 Vector Search 엔드포인트 {vector_search_endpoint_name} 에 생성 중입니다. 이 작업은 5~10분 정도 소요됩니다.")
  vsc.create_delta_sync_index(
    endpoint_name=vector_search_endpoint_name,
    index_name=vs_index_fullname,
    source_table_name=source_table_fullname,
    pipeline_type="TRIGGERED",
    primary_key="id",
    embedding_source_column='content', # 텍스트가 포함된 컬럼
    embedding_model_endpoint_name=embedding_model_name # 임베딩을 만드는 데 사용된 임베딩 엔드포인트
  )
  # 인덱스가 준비되고 모든 임베딩이 생성되고 인덱싱될 때까지 대기
  wait_for_index_to_be_ready(vsc, vector_search_endpoint_name, vs_index_fullname)
else:
  # 동기화를 트리거하여 테이블에 저장된 새 데이터로 벡터 검색 콘텐츠를 업데이트
  wait_for_index_to_be_ready(vsc, vector_search_endpoint_name, vs_index_fullname)
  vsc.get_index(vector_search_endpoint_name, vs_index_fullname).sync()

print(f"index {vs_index_fullname} on table {source_table_fullname} is ready")

# COMMAND ----------

# MAGIC %md 
# MAGIC ## 벡터값이 유사한 데이터 검색
# MAGIC
# MAGIC - 델타 테이블과 Vector Search를 동기화하면, Databricks가 자동으로 새 항목을 캡처하여 델타 라이브 테이블에 동기화합니다.
# MAGIC - 데이터 세트 크기와 모델 크기에 따라 임베딩을 시작하고 검색하는데 10초정도가 소요될 수 있습니다.
# MAGIC
# MAGIC 벡터값이 유사한 콘텐츠를 검색해 보겠습니다.
# MAGIC

# COMMAND ----------

# DBTITLE 1,Vector Search 인덱스에서 질의와 유사한 문서 검색 테스트
import mlflow.deployments
deploy_client = mlflow.deployments.get_deploy_client("databricks")

question = "SW융합산업에서 자동차 산업의 경우 구직자의 근무지는 주로 어디인가요?"

results = vsc.get_index(vector_search_endpoint_name, vs_index_fullname).similarity_search(
  query_text=question,
  columns=["created_at", "content"],
  num_results=4
)
docs = results.get('result', {}).get('data_array', [])
docs

# COMMAND ----------

# MAGIC %md
# MAGIC # 5. 검색 증강 생성(RAG) 및 Amazon Bedrock과 연계하여 챗봇 생성하기
# MAGIC
# MAGIC 이제 Databricks Vector Search 인덱스가 준비되었습니다!
# MAGIC
# MAGIC 이제 RAG를 수행하기 위해 새 RAG 모델 서비스 엔드포인트를 생성하고 배포해 보겠습니다.
# MAGIC
# MAGIC 흐름은 다음과 같습니다:
# MAGIC
# MAGIC - 사용자가 질문을 합니다.
# MAGIC - 질문이 서버리스 챗봇 RAG 엔드포인트로 전송됩니다.
# MAGIC - 엔드포인트가 임베딩을 계산하고 벡터 검색 인덱스를 활용하여 질문과 유사한 문서를 검색합니다.
# MAGIC - 엔드포인트가 해당 문서로 강화된 프롬프트를 생성합니다.
# MAGIC - 프롬프트가 Amazon Bedrock Foundation 모델 서빙 엔드포인트로 전송됩니다.
# MAGIC - 사용자에게 응답을 출력합니다. 

# COMMAND ----------

# MAGIC %md
# MAGIC <img src="https://github.com/dbrxkr/zero-to-gen-ai/blob/main/img/zero-to-genai-image03.png?raw=true" style="float: right; width: 100%; margin-left: 10px">

# COMMAND ----------

# MAGIC %md
# MAGIC ## 랭체인 리트리버(Langchain Retriever)
# MAGIC
# MAGIC 랭체인 리트리버를 만드는 것부터 시작하겠습니다. 
# MAGIC
# MAGIC 리트리버는 아래와 같은 작업을 수행 합니다 : 
# MAGIC
# MAGIC * Databricks Vector Search 인덱스가 질문의 임베딩을 계산합니다.
# MAGIC * 질문과 관련된 청크를 찾아 언어 생성 파운데이션 모델에 전송할 프롬프트를 보강합니다. 
# MAGIC
# MAGIC Databricks Langchain wrapper는 모든 기본 로직과 API 호출을 처리하여 한 단계로 쉽게 수행할 수 있게 합니다.

# COMMAND ----------

# DBTITLE 1,벡터 인덱스 이름 확인
index_name = f"{uc_catalog}.{uc_schema}.databricks_documentation_vs_index"
print(index_name)


# COMMAND ----------

# DBTITLE 1,호스트 URL 확인
host = "https://" + spark.conf.get("spark.databricks.workspaceUrl")
print(host)

# COMMAND ----------

# DBTITLE 1,벡터 검색을 위한 리트리버 생성
from databricks.vector_search.client import VectorSearchClient
from langchain_community.vectorstores import DatabricksVectorSearch
from langchain_community.embeddings import DatabricksEmbeddings

# 랭체인 모델 임베딩 테스트
# 참고: 질문 임베딩 모델은 이전 모델의 Chunk에 사용된 모델과 일치해야 합니다. 
embedding_model = DatabricksEmbeddings(endpoint=embedding_model_name)
print(f"임베딩 테스트 : {embedding_model.embed_query('대한민국의 수도는?')[:5]}...\n")

def get_retriever(persist_dir: str = None):
    
    os.environ["DATABRICKS_HOST"] = host

    # Vector Search Client 객체 생성
    vsc = VectorSearchClient(
        workspace_url=host, 
        personal_access_token=databricks_token
    )

    # Vector Search Client로부터 Vector Search Index 가져오기
    vs_index = vsc.get_index(
        endpoint_name=vector_search_endpoint_name,
        index_name=index_name
    )

    # Retriever 만들기
    vectorstore = DatabricksVectorSearch(
        index=vs_index, 
        embedding=embedding_model,
        text_column="content" 
    )

    return vectorstore.as_retriever()


# Retriever를 통해 가장 유사한 문서 4개가 반환됨
vectorstore = get_retriever()
#similar_documents = vectorstore.get_relevant_documents("자동차 기업인 General Motors의 경우 Microsoft와 협력해 개발중인 생성형AI 서비스는?")
similar_documents = vectorstore.invoke("자동차 기업인 General Motors의 경우 Microsoft와 협력해 개발중인 생성형AI 서비스는?")
print(f"\n============\n\nRelevant documents : {similar_documents[0]}")
print(f"\n============\n\nRelevant documents : {similar_documents[1]}")
print(f"\n============\n\nRelevant documents : {similar_documents[2]}")
print(f"\n============\n\nRelevant documents : {similar_documents[3]}")
print(f"\n============\n\nNumber of documents : {len(similar_documents)}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 데이터브릭스 채팅 모델을 구축하여 Amazon Bedrock 인스트럭트 파운데이션 모델 쿼리하기
# MAGIC
# MAGIC RAG 챗봇은 Amazon Bedrock을 파운데이션 모델을 사용하여 답변을 제공할 것입니다.  
# MAGIC
# MAGIC *참고: 여러 유형의 엔드포인트 또는 랭체인 모델을 사용할 수 있습니다.
# MAGIC
# MAGIC - Databricks Foundation 모델(DBRX, Llama3, Mixtral 등)
# MAGIC - 사용자가 개발하여 배포한 모델(파인튜닝한 오픈소스 모델 등)
# MAGIC - Amazon Bedrock과 같은 외부 모델 공급자 (본 실습에서 사용할 모델)

# COMMAND ----------

# DBTITLE 1,언어 생성 파운데이션 모델 생성 및 테스트
# 언어 생성 파운데이션 모델 테스트
from langchain_community.chat_models import ChatDatabricks

#chat_model = ChatDatabricks(endpoint=generative_model_name, max_tokens = 1500)
chat_model = ChatDatabricks(endpoint=generative_model_name, 
                            extra_params={"temperature": 0.1, "top_p": 0.95, "max_tokens": 1500}
                           )
print(f"파운데이션 모델 테스트 : {chat_model.invoke('자동차 기업인 General Motors의 경우 Microsoft와 협력해 개발중인 생성형AI 서비스는?')}")

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ## RAG 체인을 조립하여 완전한 챗봇을 만들기
# MAGIC
# MAGIC 이제 리트리버와 모델을 하나의 랭체인 체인에 병합해 보겠습니다.
# MAGIC
# MAGIC 챗봇이 적절한 답변을 제공할 수 있도록 커스텀 랭체인 템플릿을 사용하겠습니다.

# COMMAND ----------

# DBTITLE 1,프롬프트 템플릿을 생성하여 체인 병합
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain_community.chat_models import ChatDatabricks

TEMPLATE = """당신은 IT 소프트웨어(Software; SW) 인력 수요를 조사하는 연구원 입니다. 국내 SW인력 수요와 관련하여 산업에서 SW인력에게 원하는 역할과 요구사항을 답변하고 있습니다. 질문이 이러한 주제 중 하나와 관련이 없는 경우에는 답변을 거부해 주세요. 답을 모른다면 모른다고만 말하고 답을 지어내려고 하지 마세요. 가능한 간결하게 답변하세요. 모든 답변은 한국어로 높임말로 대답해야 합니다. 마지막으로 다음과 같은 문맥(context)을 활용하여 질문에 답하세요:
{context}
Question: {question}
Answer:
"""
prompt = PromptTemplate(template=TEMPLATE, input_variables=["context", "question"])

chain = RetrievalQA.from_chain_type(
    llm=chat_model,
    chain_type="stuff",
    retriever=get_retriever(),
    chain_type_kwargs={"prompt": prompt}
)

# COMMAND ----------

# DBTITLE 1,RAG 체인 테스트
# 만약 보내진 전체 프롬프트를 확인하고 싶다면 아래 주석을 해제하여 langchain.debug = True를 사용
# langchain.debug = True
question = {"query": "석사에게 첫번째로 요구되는 SW기술 스택 수요는 무엇인가요?"}
answer = chain.invoke(question)
print(answer)

# 엔드포인트에서 쿼리를 하고자 한다면 아래 구문 사용
"""
{
  "dataframe_split": {
    "columns": [
      "query"
    ],
    "data": [
      [
        "석사에게 첫번째로 요구되는 SW기술 스택 수요는 무엇인가요?"
      ]
    ]
  }
}
"""

# COMMAND ----------

# MAGIC %md
# MAGIC # 6. 모델을 Unity Catalog에 저장 후 모델 서빙하기

# COMMAND ----------

# MAGIC %md
# MAGIC ## Unity Catalog 레지스트리에 모델 저장하기
# MAGIC
# MAGIC 생성한 RAG 체인 모델을 Unity Catalog 스키마에 등록

# COMMAND ----------

# DBTITLE 1,생성한 RAG 체인 모델을 Unity Catalog 스키마에 등록
from mlflow.models import infer_signature
import mlflow
import langchain

mlflow.set_registry_uri("databricks-uc")
model_name = f"{uc_catalog}.{uc_schema}.my_chatbot_model"

with mlflow.start_run(run_name="my_chatbot_rag") as run:
    signature = infer_signature(question, answer)
    model_info = mlflow.langchain.log_model(
        chain,
        loader_fn=get_retriever,  # 인증을 위해 DATABRICKS_TOKEN 환경변수값을 로드하며 리트리버 로드 
        artifact_path="chain",
        registered_model_name=model_name,
        pip_requirements=[
            "mlflow==" + mlflow.__version__,
            "langchain==" + langchain.__version__,
            "databricks-vectorsearch",
        ],
        input_example=question,
        signature=signature
    )

# COMMAND ----------

# MAGIC %md 
# MAGIC ## 채팅 모델을 서버리스 모델 엔드포인트로 배포하기 
# MAGIC
# MAGIC Unity Catalog에 저장된 RAG 체인 모델을 모델 서빙으로 배포

# COMMAND ----------

# DBTITLE 1,Unity Catalog에 저장된 RAG 체인 모델을 모델 서빙으로 배포
# 서빙 엔드포인트 생성 또는 업데이트
from databricks.sdk import WorkspaceClient
from databricks.sdk.service.serving import EndpointCoreConfigInput, ServedModelInput, ServedModelInputWorkloadSize
import datetime

latest_model_version = get_latest_model_version(model_name)

w = WorkspaceClient()
endpoint_config = EndpointCoreConfigInput(
    name=serving_endpoint_name,
    served_models=[
        ServedModelInput(
            model_name=model_name,
            model_version=latest_model_version,
            workload_size=ServedModelInputWorkloadSize.SMALL,
            scale_to_zero_enabled=False,
            environment_vars={                
                "DATABRICKS_TOKEN": f"{databricks_token}",
            }
        )
    ]
)

existing_endpoint = next(
    (e for e in w.serving_endpoints.list() if e.name == serving_endpoint_name), None
)
serving_endpoint_url = f"{host}/ml/endpoints/{serving_endpoint_name}"
if existing_endpoint == None:
    print(f"엔드포인트 {serving_endpoint_url} 를 생성하고 있습니다. 엔드포인트를 배포하는데 20~25분 정도 소요됩니다. 잠시만 기다려 주세요...")
    w.serving_endpoints.create_and_wait(name=serving_endpoint_name, config=endpoint_config, timeout=datetime.timedelta(minutes=60))
else:
    print(f"엔드포인트 {serving_endpoint_url} 버전 {latest_model_version} 을 업데이트 하고 있습니다. 엔드포인트를 배포하는데 20~25분 정도 소요됩니다. 잠시만 기다려 주세요...")
    w.serving_endpoints.update_config_and_wait(served_models=endpoint_config.served_models, name=serving_endpoint_name, timeout=datetime.timedelta(minutes=60))
    
displayHTML(f'이제 모델 엔드포인트 서빙을 사용할 수 있습니다. 자세한 내용은 <a href="/ml/endpoints/{serving_endpoint_name}">모델 제공 엔드포인트 페이지</a>를 참조하세요.')

# COMMAND ----------

# MAGIC %md
# MAGIC ## 엔드포인트 배포 확인하기
# MAGIC 이제 엔드포인트가 배포되었습니다
# MAGIC 서빙 엔드포인트 [UI](#/mlflow/endpoints)에서 엔드포인트 이름을 검색하고 성능을 시각화할 수 있습니다.
# MAGIC
# MAGIC Python에서 쿼리를 실행해 보겠습니다. 

# COMMAND ----------

# DBTITLE 1,사용자 ID 확인
email = spark.sql('select current_user() as user').collect()[0]['user']
print(email)

# COMMAND ----------

# DBTITLE 1,확인된 사용자에게 카탈로그 및 스키마 권한 부여
#spark.sql(f"GRANT USAGE ON CATALOG {uc_catalog} TO `{email}`");
spark.sql(f"GRANT USAGE ON DATABASE {uc_catalog}.{uc_schema} TO `{email}`");

# COMMAND ----------

# DBTITLE 1,확인된 사용자에게 Vector Search 인덱스 조회 권한 부여
from databricks.sdk import WorkspaceClient
import databricks.sdk.service.catalog as c
WorkspaceClient().grants.update(c.SecurableType.TABLE, f"{uc_catalog}.{uc_schema}.databricks_documentation_vs_index", 
                                changes=[c.PermissionsChange(add=[c.Privilege["SELECT"]], principal=f"{email}")])

# COMMAND ----------

# DBTITLE 1,배포된 RAG 체인 엔드포인트에 쿼리 테스트
question = "헬스케어 시장 내 IT시장의 세계시장규모 성장세에 대해서 세문장 이내로 설명해 주세요."
print(w.serving_endpoints)
answer = w.serving_endpoints.query(serving_endpoint_name, inputs=[{"query": question}])
print(answer.predictions[0])


# COMMAND ----------

# MAGIC %md 
# MAGIC
# MAGIC # 7. Gradio UI를 만들기위한 코드 생성
# MAGIC - Gradio를 사용하여 UI를 만들기위한 코드를 생성
# MAGIC - 위에서 생성한 SERVING_ENDPOINT_URL을 사용하여 쿼리
# MAGIC - 미리 생성한 DATABRICKS_TOKEN을 사용하여 인증
# MAGIC - 생성된 app.py 파일을 다운로드하여 로컬 PC에서 실행하여 테스트

# COMMAND ----------

# DBTITLE 1,RAG 챗봇 테스트를 위해 Gradio를 사용하여 UI 생성
# 관련 Python 패키지 로드.
import gradio as gr
import os
import requests
import pandas as pd
import json


# 환경 변수 설정. 실제 환경에서는 OS레벨에서 환경 변수로 설정 하여 보안을 강화해야합니다.
os.environ['SERVING_ENDPOINT_URL'] = f"{host}/serving-endpoints/{serving_endpoint_name}/invocations"
os.environ['DATABRICKS_TOKEN'] = f"{databricks_token}"


# 질의를 JSON 형태로 변환.
def create_tf_serving_json(data):
    return {'inputs': {name: data[name] for name in data.keys()} if isinstance(data, dict) else data}


# Databricks RAG 서빙 엔드포인트에 쿼리.
def rag_invoke(dataset):
    url = f'{os.environ.get("SERVING_ENDPOINT_URL")}'
    headers = {'Authorization': f'Bearer {os.environ.get("DATABRICKS_TOKEN")}', 'Content-Type': 'application/json'}
    ds_dict = {'dataframe_split': dataset.to_dict(orient='split')} if isinstance(dataset, pd.DataFrame) else create_tf_serving_json(dataset)
    print(ds_dict)
    data_json = json.dumps(ds_dict, allow_nan=True)
    response = requests.request(method='POST', headers=headers, url=url, data=data_json)
    if response.status_code != 200:
        raise Exception(f'Request failed with status {response.status_code}, {response.text}')
    return response.json()


# Gradio에서 입력받은 질의에 대한 응답을 생성. message는 사용자의 질의 메시지이며, history는 과거 대화이력, additional_input_info는 additional_inputs의 메시지.
def answer_chat(message, history, additional_input_info):
     
    if len(history) == 0 :
        query_string = f"""{message}. 
                    {additional_input_info}."""
    else :
        query_string = f"""{message}.
                    {additional_input_info}. 
                    참고로 이전의 질문과 답변은 다음과 같습니다. 질문 : {history[len(history)-1][0]}. 답변 : {history[len(history)-1][1]}"""
        
    query_json = {"query": [query_string]}    
    rag_response = rag_invoke(query_json)    
    answer = rag_response['predictions'][0]

    print(query_json)
    print(answer)

    return answer


# Gradio Blocks 구문
with gr.Blocks(theme=gr.themes.Soft()) as demo:

    # 버튼 스타일을 위한 CSS 추가
    gr.HTML(
        """
        <style>   
        #chatbot {
            border: 2px solid #787efa !important;
            background-color: white !important;
            color: black !important;            
        }            
        #submit-btn {
            background-color: #787efa !important;
            color: white !important;
            border: none !important;
            padding: 10px 20px !important;
            font-size: 16px !important;
            cursor: pointer !important;
        }
        #submit-btn:hover {
            background-color: #5c64fa !important;
        }
        #message-input {
            border: 2px solid #787efa !important;
            background-color: white !important;
            color: black !important;   
        }        
        </style>
        """
    )

    gr.Markdown(
        """
        # Zero to GenAI 워크샵 RAG 봇
        이 챗봇은 IT 소프트웨어 인력 수요 정보를 검색하는 LLM RAG 데모 예제입니다. <br>
        Databricks Vector Search 데이터베이스에 저장된 문서에서 유사한 문서를 찾아서 답변합니다.
        """
    )

    # 대화 목록
    chatbot = gr.Chatbot(height=400, elem_id="chatbot")
    
    # 대화 입력
    

    # 전송 버튼
    with gr.Row():
        message_input = gr.Textbox(placeholder="대화를 입력하세요.", container=False, scale=7, elem_id="message-input")
        submit_btn = gr.Button("메시지 전송", elem_id="submit-btn")

    # 추가 프롬프트 입력
    with gr.Row():
        additional_input_info = gr.Textbox("", label="추가 프롬프트 입력")
    
    # 예제 입력
    examples = gr.Examples(
        examples=[
            ["자동차 기업인 General Motors가 Microsoft와 협력해 개발중인 생성형AI 서비스는?"], 
            ["헬스케어 시장 내 IT시장의 세계시장규모 성장세에 대해서 세문장 이내로 설명해 주세요."],
            ["소프트웨어진흥법 제2조 6의 내용은 무엇인가요?"],
            ["석사에게 첫번째로 요구되는 SW기술 스택 수요는 무엇인가요?"],
            ["SW융합산업에서 자동차 산업의 경우 구직자의 근무지는 주로 어디인가요?"],
        ],
        inputs=[message_input]
    )

    # 전체대화 삭제, 이전대화 삭제 버튼
    with gr.Row():
        clear_btn = gr.Button("전체대화 삭제")
        undo_btn = gr.Button("이전대화 삭제")

    
    def handle_submit(message, history, additional_input_info):
        response = answer_chat(message, history, additional_input_info)
        history.append((message, response))
        return history
    
    message_input.submit(
        fn=handle_submit,
        inputs=[message_input, chatbot, additional_input_info],
        outputs=[chatbot]
    )

    submit_btn.click(
        fn=handle_submit,
        inputs=[message_input, chatbot, additional_input_info],
        outputs=[chatbot]
    )
    
    clear_btn.click(lambda: [], None, chatbot)
    undo_btn.click(lambda history: history[:-1], chatbot, chatbot)

#demo.launch(server_name="localhost", server_port=8080)


# COMMAND ----------

# DBTITLE 1,Gradio로 개발된 UI를 Spark Driver 노드에 임시로 구동하여 테스트
from dbtunnel import dbtunnel
dbtunnel.gradio(demo).run()

# COMMAND ----------

# MAGIC %md-sandbox
# MAGIC # 8. 교육 내용 초기화
# MAGIC <br/>
# MAGIC <div style="background-color: #fad7d0; padding: 15px; boarder: 30px; ">
# MAGIC   <strong>⛔ 주 의</strong><br/>
# MAGIC   - 아래셀은 교육 컨텐츠를 초기화 하는 스크립트 입니다<br/>
# MAGIC   - 아래 주석을 해제하고 실행하면 실습 스키마를 완전히 삭제합니다<br/>  
# MAGIC </div>

# COMMAND ----------

# ⛔ 아래 주석을 해제하고 실행하면 실습 스키마를 완전히 삭제합니다 ⛔
# cleanup_demo(catalog, db, serving_endpoint_name, f"{uc_catalog}.{uc_schema}.databricks_documentation_vs_index")

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC %md
# MAGIC <img src="https://github.com/dbrxkr/zero-to-gen-ai/blob/main/img/zero-to-genai-footer-black.png?raw=true" style="float: right; width: 100%; margin-left: 10px">
# MAGIC
# MAGIC

# COMMAND ----------

# MAGIC %environment
# MAGIC "client": "1"
# MAGIC "base_environment": ""
