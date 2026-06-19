import requests
import json
import logging
from typing import Dict, Any, Optional

class CVDLlamaAdvisor:
    """
    Local Llama advisor for personalized CVD and environmental health advice
    """
    
    def __init__(self, model_name: str = "llama3.2:3b"):
        self.ollama_url = "http://localhost:11434/api/generate"
        self.model = model_name
        self.timeout = 15  # seconds
        
        # Setup logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
    def get_environmental_advice(self, risk_level: str, environmental_data: Dict, user_data: Dict) -> str:
        """
        Generate personalized environmental and lifestyle advice based on CVD risk and location
        """
        try:
            prompt = self._build_advice_prompt(risk_level, environmental_data, user_data)
            advice = self._query_llama(prompt)
            
            # Validate and clean response
            if advice and len(advice.strip()) > 20:
                return self._clean_response(advice)
            else:
                return self._get_fallback_advice(risk_level, environmental_data)
                
        except Exception as e:
            self.logger.error(f"Error generating LLM advice: {str(e)}")
            return self._get_fallback_advice(risk_level, environmental_data)
    
    def _build_advice_prompt(self, risk_level: str, env_data: Dict, user_data: Dict) -> str:
        """
        Build a comprehensive prompt for the LLM based on user context
        """
        borough = env_data.get('borough', 'London')
        pm25 = env_data.get('pm25', 0)
        no2 = env_data.get('no2', 0)
        age = user_data.get('Age', 'unknown')
        
        # Determine pollution level
        pollution_level = "high" if pm25 > 15 or no2 > 50 else "moderate" if pm25 > 10 or no2 > 40 else "low"
        
        # Avoid "medical advice" and focus on practical, environmental, and lifestyle tips
        prompt = (
            f"You are a friendly health and environment advisor for London residents.\n"
            f"Profile:\n"
            f"- Age: {age}\n"
            f"- Borough: {borough}\n"
            f"- Air Quality: PM2.5 = {pm25:.1f} μg/m³, NO2 = {no2:.1f} μg/m³ ({pollution_level} pollution)\n"
            f"- Cardiovascular risk: {risk_level.lower()}\n\n"
            "Please give 3 practical, encouraging tips to help this person reduce their cardiovascular risk and stay healthy, "
            "focusing on environmental and lifestyle actions they can take in London. "
            "Do not provide medical advice or disclaimers—just helpful, everyday suggestions."
        )

        return prompt
    
    def _query_llama(self, prompt: str) -> Optional[str]:
        """
        Query the local Ollama instance
        """
        try:
            payload = {
                "model": self.model,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": 0.7,
                    "top_p": 0.9,
                    "num_predict": 150  # Limit response length
                }
            }
            
            self.logger.info(f"Querying Ollama with model: {self.model}")
            
            response = requests.post(
                self.ollama_url, 
                json=payload,
                timeout=self.timeout,
                headers={'Content-Type': 'application/json'}
            )
            
            if response.status_code == 200:
                result = response.json()
                return result.get("response", "").strip()
            else:
                self.logger.error(f"Ollama request failed with status: {response.status_code}")
                return None
                
        except requests.exceptions.ConnectionError:
            self.logger.error("Cannot connect to Ollama. Is it running on localhost:11434?")
            return None
        except requests.exceptions.Timeout:
            self.logger.error("Ollama request timed out")
            return None
        except Exception as e:
            self.logger.error(f"Unexpected error querying Ollama: {str(e)}")
            return None
    
    def _clean_response(self, response: str) -> str:
        """
        Clean and format the LLM response
        """
        # Remove any unwanted prefixes or suffixes
        response = response.strip()
        
        # Ensure it doesn't exceed reasonable length
        if len(response) > 500:
            response = response[:500] + "..."
        
        return response
    
    def _get_fallback_advice(self, risk_level: str, env_data: Dict) -> str:
        """
        Provide fallback advice when LLM is unavailable
        """
        borough = env_data.get('borough', 'your area')
        pm25 = env_data.get('pm25', 0)
        no2 = env_data.get('no2', 0)
        
        base_advice = {
            'Low Risk': [
                f"1. Continue your healthy lifestyle while being mindful of air quality in {borough}.",
                "2. Exercise in London's green spaces like Hyde Park or Hampstead Heath when possible.",
                "3. Monitor London Air Quality app before outdoor activities and maintain regular check-ups."
            ],
            'Moderate Risk': [
                f"1. Exercise indoors or in parks when air pollution is high in {borough}.",
                "2. Consider cycling on London's Cycle Superhighways to avoid traffic pollution.",
                "3. Increase cardiovascular exercise while avoiding busy roads during peak hours."
            ],
            'High Risk': [
                f"1. Limit outdoor exercise when PM2.5 > 15 μg/m³ in {borough}.",
                "2. Consult your GP about air pollution's impact on your cardiovascular health.",
                "3. Use air purifiers at home and consider relocation if air quality is consistently poor."
            ]
        }
        
        advice_list = base_advice.get(risk_level, base_advice['Moderate Risk'])
        return " ".join(advice_list)
    
    def check_ollama_availability(self) -> bool:
        """
        Check if Ollama is running and the model is available
        """
        try:
            # Test connection with a simple query
            test_response = self._query_llama("Hello")
            return test_response is not None
        except:
            return False
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        Get information about the current model
        """
        try:
            response = requests.get("http://localhost:11434/api/tags", timeout=5)
            if response.status_code == 200:
                models = response.json().get("models", [])
                current_model = next((m for m in models if m["name"] == self.model), None)
                return {
                    "model_name": self.model,
                    "available": current_model is not None,
                    "ollama_running": True
                }
        except:
            pass
        
        return {
            "model_name": self.model,
            "available": False,
            "ollama_running": False
        }