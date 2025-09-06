const API_BASE_URL =
    import.meta.env.VITE_API_BASE_URL || "http://localhost:8000";

async function handleJsonResponse(response) {
    if (!response.ok) {
        let errorBody;
        try {
            errorBody = await response.json();
        } catch (_) {
            errorBody = { detail: await response.text() };
        }
        const error = new Error("Request failed");
        error.status = response.status;
        error.body = errorBody;
        throw error;
    }
    return response.json();
}

const apiService = {
    async createUser(payload) {
        const res = await fetch(`${API_BASE_URL}/users/`, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify(payload)
        });
        return handleJsonResponse(res);
    }
};

export default apiService;