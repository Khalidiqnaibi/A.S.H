# episode_ranker.py

import time


class EpisodeRanker:

    def rank(self, scored_episodes):
        """
        scored_episodes: List[(similarity_score, Episode)]
        """

        now = time.time()
        ranked = []

        for sim, ep in scored_episodes:

            # recency weight (recent = higher)
            age = now - ep.timestamp
            recency_score = 1 / (1 + age / 86400)  # 1-day decay

            final_score = (
                0.5 * sim +
                0.3 * recency_score +
                0.2 * ep.importance
            )

            ranked.append((final_score, ep))

        ranked.sort(reverse=True, key=lambda x: x[0])