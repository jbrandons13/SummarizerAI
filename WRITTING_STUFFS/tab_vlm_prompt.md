# VLM-as-Judge Evaluation Prompt

**Instructions:**
1. Open the VLM interface (e.g., Google AI Studio for Gemini 1.5 Pro/Flash).
2. Upload the final generated summary video (`video_daca.mp4` or `video_fixed_w02.mp4`).
3. Paste the following prompt:

---

**Prompt:**
You are an expert video editor and evaluator. I have provided a generated video summary of an educational video. Please evaluate the video on the following three criteria using a scale of 1 to 5 (1 = Poor, 5 = Excellent).

**Rubric:**
1. **Faithfulness to the source:** Does the video accurately and clearly convey the main content and factual claims of the educational topic? (Score 1-5)
2. **Visual coherence:** Is the visual style consistent throughout? If there are recurring concepts (e.g., a specific character or environment), are they recognizable across different shots? (Score 1-5)
3. **Narration-visual alignment:** Do the visuals on screen match what the voiceover narration is saying at any given time? (Score 1-5)

Please provide a short justification for each score, followed by the final numerical scores in this format:
Faithfulness: [Score]
Coherence: [Score]
Alignment: [Score]

---

**Scores (Placeholder to be filled after running the VLM):**

**Video: Geology DACA**
- Faithfulness: TBD
- Coherence: TBD
- Alignment: TBD

**Video: Ecology DACA**
- Faithfulness: TBD
- Coherence: TBD
- Alignment: TBD
