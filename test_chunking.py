import time
import json
import urllib.request
import urllib.parse
import urllib.error

BASE_URL = "http://localhost:8000"

def post_multipart(url, file_path, params):
    from uuid import uuid4
    
    boundary = uuid4().hex
    
    if params:
        url = url + "?" + urllib.parse.urlencode(params)
    
    with open(file_path, "rb") as f:
        file_data = f.read()

    body = []
    body.append(f"--{boundary}\r\n".encode())
    body.append(f"Content-Disposition: form-data; name=\"file\"; filename=\"test_data.csv\"\r\n".encode())
    body.append(b"Content-Type: text/csv\r\n\r\n")
    body.append(file_data)
    body.append(b"\r\n")
    body.append(f"--{boundary}--\r\n".encode())
    
    payload = b"".join(body)
    
    req = urllib.request.Request(url, data=payload, method="POST")
    req.add_header("Content-Type", f"multipart/form-data; boundary={boundary}")
    
    try:
        with urllib.request.urlopen(req) as response:
            return response.status, json.loads(response.read().decode())
    except urllib.error.HTTPError as e:
        return e.code, e.read().decode()

def post_json(url, data):
    payload = json.dumps(data).encode("utf-8")
    req = urllib.request.Request(url, data=payload, method="POST")
    req.add_header("Content-Type", "application/json")
    try:
        with urllib.request.urlopen(req) as response:
            return response.status, json.loads(response.read().decode())
    except urllib.error.HTTPError as e:
        return e.code, e.read().decode()

def test_chunking():
    print("Testing chunking strategies on /ingest/csv...")
    strategies = ["none", "character", "word", "recursive"]
    for strategy in strategies:
        print(f"\n--- Testing strategy: {strategy} ---")
        params = {"chunking_strategy": strategy, "chunk_size": 150, "chunk_overlap": 30}
        
        status, response = post_multipart(f"{BASE_URL}/api/v1/ingest/csv", "test_data.csv", params)
        print(f"Ingest Response ({status}): {response}")
        
        time.sleep(2)
        
        query = {"query": "lightweight", "limit": 3, "prefetch_limit": 6}
        status, search_res = post_json(f"{BASE_URL}/api/v1/search", query)
        print(f"Search Response ({status}):")
        if status == 200:
            for idx, res in enumerate(search_res.get("results", [])):
                payload = res.get("payload", {})
                chunk_meta = payload.get('chunk_metadata', {})
                meta_str = f"[{chunk_meta.get('strategy')}|{chunk_meta.get('size')}|{chunk_meta.get('overlap')}]" if chunk_meta else ""
                print(f"  Result {idx+1}: Score={res.get('score')} - ID={payload.get('id')} - Meta={meta_str} - Desc={payload.get('description', '')[:80]}...")
        else:
            print(search_res)

if __name__ == "__main__":
    test_chunking()
