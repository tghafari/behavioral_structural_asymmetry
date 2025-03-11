def analyze_data(fpath):
    """Analyze single subject data from CSV file."""
    try:
        data = pd.read_csv(fpath)
        data = data[data['State'] == 1]
        contrasts = np.unique(data['Contrast'])

        # Preallocate Results
        results = np.stack((
            contrasts,
            np.zeros(np.shape(contrasts)[0]),
            np.zeros(np.shape(contrasts)[0]),
            np.zeros(np.shape(contrasts)[0])
        ), axis=1)

        for contrast in contrasts:
            contrast_data = data[data['Contrast'] == contrast]
            contrast_trials_all = len(contrast_data)

            # Process right attention trials
            contrast_attention_right = contrast_data[contrast_data['Attention_Direction'] == 'Right']
            contrast_trials_right = len(contrast_attention_right)

            right_corrects = contrast_attention_right[
                ((contrast_attention_right['Answer'] == 'LeftShift') &
                 (contrast_attention_right['Target_Oriention'] == 45)) |
                ((contrast_attention_right['Answer'] == 'RightShift') &
                 (contrast_attention_right['Target_Oriention'] == -45))
            ]
            right_correct_percent = len(
                right_corrects) / contrast_trials_right if contrast_trials_right > 0 else 0

            # Process left attention trials
            contrast_attention_left = contrast_data[contrast_data['Attention_Direction'] == 'Left']
            contrast_trials_left = len(contrast_attention_left)

            left_corrects = contrast_attention_left[
                ((contrast_attention_left['Answer'] == 'LeftShift') &
                 (contrast_attention_left['Target_Oriention'] == 45)) |
                ((contrast_attention_left['Answer'] == 'RightShift') &
                 (contrast_attention_left['Target_Oriention'] == -45))
            ]
            left_correct_percent = len(
                left_corrects) / contrast_trials_left if contrast_trials_left > 0 else 0

            # Update results
            idx = np.where(results[:, 0] == contrast)[0][0]
            results[idx, 1] = right_correct_percent
            results[idx, 2] = left_correct_percent
            results[idx, 3] = (len(right_corrects) +
                               len(left_corrects)) / contrast_trials_all

        contrast_table = pd.DataFrame(
            data=results,
            columns=["Contrast", "Right_Correct_Percent",
                     "Left_Correct_Percent", "All_Correct_Percent"]
        ).set_index(['Contrast'])

        return contrast_table