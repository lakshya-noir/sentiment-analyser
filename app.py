import os 
import argparse # for cli aruguments
import pandas as pd # to save results in a csv file
from transformers import pipeline # hugging face model for sentiment analysis
from colorama import Fore, Style # to make the cli output look more appealing

sentiment_pipeline = pipeline("sentiment-analysis") # this is the ai model we're working with

def analyse_text(text):
	result = sentiment_pipeline(text)[0]
	label = result['label']
	score = result['score']
	if label == "POSITIVE":
		label_colored = Fore.GREEN + "POSITIVE 😎" + Style.RESET_ALL
	elif label == "NEGATIVE":
		label_colored = Fore.RED + "NEGATIVE 😩" + Style.RESET_ALL
	else:
		label_colored = Fore.YELLOW + "Neutral 😴" + Style.RESET_ALL

	print(f"Text : {text}\n")
	print(f"Sentiment : {label_colored} (Confidence : {score:.2f})")
	
	return{"text":text,"sentiment":label,"score":score}

def main():
	parser = argparse.ArgumentParser(description = "Your Sentiment Analyser")
	parser.add_argument("--text", type=str, help="Input some text to analyse")
	parser.add_argument("--file", type=str, help="Add path to file with multiple lines to analyse")
	args = parser.parse_args()

	history = []

	if args.text: # runs when args.text != NONE
		history.append(analyse_text(args.text))
	if args.file: # runs when args.file != NONE
		with open(args.file, "r") as f:
			for line in f:
				line = line.strip() # to remove white spaces
				if line: # to skip empty lines
					history.append(analyse_text(line))

	if history: # runs when the list is not empty
		df = pd.DataFrame(history)		
		if os.path.exists("results.csv"):
			df.to_csv("results.csv", index=False, mode='a', header=False)
		else:
			df.to_csv("results.csv", index=False)
		print(Fore.CYAN + "Results have been saved to results.csv✅\n" + Style.RESET_ALL)

if __name__ == "__main__":
	main()
