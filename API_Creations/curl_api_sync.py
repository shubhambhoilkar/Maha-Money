from fastapi import FastAPI, HTTPException
import requests

app = FastAPI()

# Define the Headers

HEADERS = {
    "Authorization": "Basic a2FyYWY6a2FyYWY=",
    "Accept": "application/json"
}

@app.get("/proxy-profile/")
def proxy_profile(user_id: str):
    TARGET_URL = f"http://13.203.110.196:8181/cxs/profiles/{user_id}"
    try:
        response = requests.get(TARGET_URL, headers=HEADERS)
        response.raise_for_status()  # Will raise an error for 4xx/5xx status codes
        print(response.json())
        return response.json()
    
    except requests.HTTPError as http_err:
        raise HTTPException(status_code=response.status_code, detail=str(http_err))
    
    except requests.RequestException as req_err:
        raise HTTPException(status_code=500, detail=f"Request failed: {req_err}")

if __name__ == "__main__":
    import uvicorn 
    uvicorn.run("curl_api_sync:app", host= "localhost", port = 9900 , reload= True)
#   uvicorn.run("curl_api_sync:app", host= "0.0.0.0", port = 9900 , reload= True)