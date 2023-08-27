@app.route('/')
def home():
    return render_template('index.html')

@app.route('/analyze_sentiment', methods=['POST'])
def analyze_sentiment():
    text_input = request.form['text_input']
    
    sentiment_scores = sid.polarity_scores(text_input)
    polarity = sentiment_scores['compound']

    if polarity > 0.6:
        sentiment = 'Positive'
        sentiment_text = 'This text is positive!'
    elif polarity < 0.1:
        sentiment = 'Negative'
        sentiment_text = 'This text is negative.'
    else:
        sentiment = 'Neutral'
        sentiment_text = 'This text is neutral.'

    return render_template('index.html', sentiment_scores=sentiment_scores,
                           sentiment=sentiment, sentiment_text=sentiment_text)

@app.route('/clean_text', methods=['POST'])
def clean_text():
    clean_input = request.form['clean_input']

    cleaned_text = cleaner(clean_input)
    preprocessed_text = preprocess_text(cleaned_text)

    return render_template('index.html', cleaned_text=cleaned_text,
                           preprocessed_text=preprocessed_text)

@app.route('/analyze_csv', methods=['POST'])
def analyze_csv():
    file = request.files['file']
    if file:
        df = pd.read_csv(file)

        # Define the score and analyze functions
        def score(x):
            # ... (same as in the Streamlit code)

        def analyze(x):
            # ... (same as in the Streamlit code)

        df['score'] = df['tweets'].apply(score)
        df['analysis'] = df['score'].apply(analyze)

        csv = df.to_csv(index=False)
        return send_file(csv, mimetype='text/csv', as_attachment=True, download_name='sentiment.csv')
