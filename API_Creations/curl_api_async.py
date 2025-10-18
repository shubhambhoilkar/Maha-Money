from fastapi import FastAPI, HTTPException
import httpx

app = FastAPI()

# Authorization header and Accept header as specified in the cURL
HEADERS = {
    "Authorization": "Basic a2FyYWY6a2FyYWY=",
    "Accept": "application/json"
}

@app.get("/proxy-profile/")
async def proxy_profile(user_id: str):
    async with httpx.AsyncClient() as client:
        try:
        #    TARGET_URL = "http://13.203.110.196:8181/cxs/profiles/13709473-4b3e-40cc-9b42-9785063cf1e5"
            TARGET_URL = f"http://13.203.110.196:8181/cxs/profiles/{user_id}"

            # Make the GET request to the target URL
            response = await client.get(TARGET_URL, headers=HEADERS)
            response.raise_for_status()
            return response.json()

        except httpx.HTTPStatusError as http_err:
            raise HTTPException(status_code=response.status_code, detail=str(http_err))

        except httpx.RequestError as req_err:
            raise HTTPException(status_code=500, detail=f"Request failed: {req_err}")