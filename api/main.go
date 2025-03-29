package main

import (
	"bytes"
	"encoding/json"
	"fmt"
	"log"
	"net/http"
	"os"
	"os/exec"

	"github.com/gin-gonic/gin"
)

// PredictionResponse represents the response format from the Python script.
type PredictionResponse struct {
	PredictedDigit int                `json:"predicted_digit"`
	Probabilities  map[string]float64 `json:"probabilities"`
}

// classifyImage handles image classification by processing an uploaded image,
// sending it to the Python inference script, and returning the prediction results.
func classifyImage(c *gin.Context) {
	file, err := c.FormFile("image")
	if err != nil {
		c.JSON(http.StatusBadRequest, gin.H{"error": "Invalid file upload"})
		return
	}

	tempFile := "temp_image.png"
	if err := c.SaveUploadedFile(file, tempFile); err != nil {
		c.JSON(http.StatusInternalServerError, gin.H{"error": "Failed to save image"})
		return
	}

	cmd := exec.Command("python3", "inference.py", tempFile)

	var out bytes.Buffer
	cmd.Stdout = &out
	cmd.Stderr = os.Stderr

	if err := cmd.Run(); err != nil {
		c.JSON(http.StatusInternalServerError, gin.H{"error": "Failed to run inference"})
		return
	}

	var response PredictionResponse
	if err := json.Unmarshal(out.Bytes(), &response); err != nil {
		c.JSON(http.StatusInternalServerError, gin.H{"error": "Failed to parse prediction result"})
		return
	}

	c.JSON(http.StatusOK, response)

	os.Remove(tempFile)
}

// main initializes the Gin router and starts the server to handle image classification requests.
func main() {
	r := gin.Default()
	r.POST("/classify", classifyImage)

	port := "8080"
	fmt.Printf("Server running on port %s...\n", port)
	log.Fatal(r.Run(":" + port))
}
