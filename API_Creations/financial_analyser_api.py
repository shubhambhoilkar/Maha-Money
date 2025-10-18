from fastapi import FastAPI, HTTPException
import requests
from financial_profile_analyzer import analyze_customer_profile

app = FastAPI(title= "Financial Analyser Middleware API.")

HEADERS = {
    "Authorization": "Basic a2FyYWY6a2FyYWY=",
    "Accept": "application/json"
}

@app.get("/proxy-profile/")
def proxy_profile(user_id: str , include_raw : bool = False):
    TARGET_URL = f"http://13.203.110.196:8181/cxs/profiles/{user_id}"
    try:
        #Step 1: GEt  the Raw Profile Data in JSON format

        response = requests.get(TARGET_URL, headers=HEADERS)
        response.raise_for_status()  # Will raise an error for 4xx/5xx status codes
        profile_data = response.json()
        print(profile_data)

        #Step 2:  analyse the profile
        analysis = analyze_customer_profile(profile_data)

        #Step 3: Return Combined Response
        return {
            "status" : "Success",
            "user_id" : user_id,
            "analysis" : analysis,
            "raw_profile" : profile_data  if include_raw else None
        }

    #    return response.json()
    
    except requests.HTTPError as http_err:
        raise HTTPException(status_code=response.status_code, detail=str(http_err))
    
    except requests.RequestException as req_err:
        raise HTTPException(status_code=500, detail=f"Request failed: {req_err}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Analyse failed: {str(e)}")

if __name__ == "__main__":
    import uvicorn 
    uvicorn.run("financial_analyser_api:app", host= "localhost", port = 9918 , reload= True)
#   uvicorn.run("curl_api_sync:app", host= "0.0.0.0", port = 9900 , reload= True)