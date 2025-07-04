document.getElementById("loanForm").addEventListener("submit", async (e) => {
  e.preventDefault();
  
  // DOM Elements
  const loanId = document.getElementById("loanId").value.trim();
  const predictBtn = document.getElementById("predictBtn");
  const resultDiv = document.getElementById("result");
  const statusSpan = document.getElementById("status");
  const confidenceSpan = document.getElementById("confidence");
  const errorDiv = document.getElementById("error") || createErrorElement();

  // Clear previous states
  resetUI(predictBtn, resultDiv, errorDiv);

  // Validation
  if (!loanId) {
    showError("Please enter a valid Loan ID", errorDiv);
    return;
  }

  // Set loading state
  setLoadingState(predictBtn, true);

  try {
    const response = await makePredictionRequest(loanId);
    const data = await processResponse(response);
    
    // Update UI with results
    updateResultsUI(data, statusSpan, confidenceSpan, resultDiv);
    
  } catch (error) {
    handlePredictionError(error, errorDiv);
  } finally {
    setLoadingState(predictBtn, false);
  }
});

// ==================== Helper Functions ====================

async function makePredictionRequest(loanId) {
  const API_URL = "http://localhost:8000/api/predict/";
  
  const response = await fetch(API_URL, {
    method: "POST",
    headers: { 
      "Content-Type": "application/json",
      // Add if using authentication:
      // "Authorization": `Bearer ${localStorage.getItem('access_token')}`
    },
    body: JSON.stringify({ loan_id: loanId }),
  });

  if (!response.ok) {
    const errorData = await response.json().catch(() => ({}));
    throw new Error(
      errorData.error || 
      `Request failed with status ${response.status}`
    );
  }

  return response;
}

async function processResponse(response) {
  const data = await response.json();
  
  // Validate response structure
  if (!data.status || data.confidence === undefined) {
    throw new Error("Invalid response format from server");
  }
  
  return {
    status: data.status.toUpperCase(),
    confidence: data.confidence.toFixed(2),
    success: data.status.toLowerCase() === "approved"
  };
}

function updateResultsUI(data, statusEl, confidenceEl, containerEl) {
  statusEl.textContent = data.status;
  confidenceEl.textContent = `${data.confidence}%`;
  
  containerEl.className = `result ${data.success ? 'approved' : 'rejected'}`;
  containerEl.classList.remove("hidden");
}

function handlePredictionError(error, errorDiv) {
  console.error("Prediction Error:", error);
  
  const userMessage = error.message.includes("failed with status 401")
    ? "Please login to make predictions" 
    : error.message;
    
  showError(userMessage, errorDiv);
}

// ==================== UI Utilities ====================

function resetUI(buttonEl, resultEl, errorEl) {
  buttonEl.textContent = "Check Approval";
  resultEl.classList.add("hidden");
  errorEl.classList.add("hidden");
}

function setLoadingState(buttonEl, isLoading) {
  buttonEl.disabled = isLoading;
  buttonEl.textContent = isLoading ? "Predicting..." : "Check Approval";
}

function showError(message, errorDiv) {
  errorDiv.textContent = message;
  errorDiv.classList.remove("hidden");
}

function createErrorElement() {
  const div = document.createElement("div");
  div.id = "error";
  div.className = "error hidden";
  document.getElementById("loanForm").appendChild(div);
  return div;
}