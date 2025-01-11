from youtube_transcript_api import YouTubeTranscriptApi
from transformers import pipeline

def summarize_youtube_captions(video_id):
    """
    Extracts captions, summarizes using BART, and adds descriptive wording about the video's content.
    """
    try:
        transcript = YouTubeTranscriptApi.get_transcript(video_id, languages=["en"])
    except Exception as e:
        print(f"Error getting transcript: {e}")
        return None

    captions = " ".join([segment['text'] for segment in transcript if segment['text'].lower() != "foreign"])

    if not captions.strip():
        return "No captions found for this video."

    summarizer = pipeline("summarization", model="facebook/bart-large-cnn")

    max_chunk_length = 1024
    chunks = [captions[i:i + max_chunk_length] for i in range(0, len(captions), max_chunk_length)]

    summaries = []
    for chunk in chunks:
        if len(chunk.strip()) > 100:
            try:
                summary = summarizer(chunk, max_length=130, min_length=30, do_sample=False)
                summaries.append(summary[0]['summary_text'])
            except Exception as e:
                print(f"Error during summarization: {e}")
                return "A problem occurred during summarization."

    final_summary = " ".join(summaries)

    if not final_summary:
        return "Could not generate a meaningful summary."

    # Descriptive wording about the video's content
    description = "The video script "
    if "explains" in final_summary.lower() or "defining" in final_summary.lower() or "what is" in final_summary.lower():
        description += "explains/defines "
    elif "discusses" in final_summary.lower() or "talking about" in final_summary.lower():
        description += "discusses "
    elif "shows" in final_summary.lower() or "demonstrates" in final_summary.lower():
        description += "shows/demonstrates "
    elif "using" in final_summary.lower() or "with" in final_summary.lower():
        description += "using/with "
    else:
        description += "talks about "

    description += final_summary + "."

    return description

if __name__ == "__main__":
    video_id = "bxuYDT-BWaI"  # Example video ID (APIs explained)
    summary = summarize_youtube_captions(video_id)
    if summary:
        print("\nSummary:")
        print(summary)
    else:
        print("Failed to generate summary.")