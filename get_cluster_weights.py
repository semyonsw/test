import urllib.request
import json
import ssl


def get_cluster_weights():
    url = "https://haroutonex.app.n8n.cloud/webhook/get-cluster-weights"

    # Optional: ignore SSL certificate errors if needed, though standard contexts are usually fine
    ctx = ssl.create_default_context()

    try:
        print(f"Making GET request to {url}...")
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, context=ctx, timeout=30) as response:
            status_code = response.getcode()
            print(f"Status Code: {status_code}")

            response_data = response.read().decode("utf-8")

            try:
                # Try to parse JSON output for better readability
                parsed_json = json.loads(response_data)
                print("Response JSON:")
                print(json.dumps(parsed_json, indent=2, ensure_ascii=False))
            except json.JSONDecodeError:
                print("Response Body:")
                print(response_data)

    except urllib.error.HTTPError as e:
        print(f"HTTP Error: {e.code}")
        print("Response Body:")
        print(e.read().decode("utf-8"))
    except urllib.error.URLError as e:
        print(f"Failed to reach the server. Reason: {e.reason}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")


if __name__ == "__main__":
    get_cluster_weights()
