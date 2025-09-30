


// helper function to call from each metric page
function sendMetric(result) {

  console.log("Sending metric:", result);

  //send the request to the log
  fetch("http://localhost:5050/log", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      result: result,
      timestamp: Date.now()
    })
  }).catch(err => console.error("Metric log failed:", err));
}