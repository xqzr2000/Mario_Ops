"""
cloud/monitoring.py -- CloudWatch custom metrics for MarioOps.

Same no-op philosophy as storage.py: unless MARIOOPS_CLOUDWATCH=true
(and boto3 is available), publish_episode() does nothing, so local CPU
test runs pay zero AWS cost and need zero credentials.

Metrics land in the MARIOOPS_CW_NAMESPACE namespace with a RunId
dimension, which is what infra/cloudwatch-dashboard.json plots.
"""

from config import (
    CLOUDWATCH_ENABLED,
    CLOUDWATCH_NAMESPACE,
    RUN_ID,
    AWS_REGION,
)


class TrainingMetrics:
    def __init__(self):
        self.namespace = CLOUDWATCH_NAMESPACE
        self.run_id = RUN_ID
        self._client = None

        if not CLOUDWATCH_ENABLED:
            self.enabled = False
            return

        try:
            import boto3
            self._client = boto3.client("cloudwatch", region_name=AWS_REGION)
            self.enabled = True
        except Exception as exc:
            print(f"[cloudwatch] disabled (boto3 unavailable: {exc})")
            self.enabled = False

    def publish_episode(self, episode, reward, loss, epsilon, steps,
                        memory_length) -> None:
        if not self.enabled:
            return

        dims = [{"Name": "RunId", "Value": self.run_id}]
        data = [
            {"MetricName": "EpisodeReward", "Value": float(reward),
             "Unit": "None", "Dimensions": dims},
            {"MetricName": "Epsilon", "Value": float(epsilon),
             "Unit": "None", "Dimensions": dims},
            {"MetricName": "EpisodeSteps", "Value": float(steps),
             "Unit": "Count", "Dimensions": dims},
            {"MetricName": "ReplayMemorySize", "Value": float(memory_length),
             "Unit": "Count", "Dimensions": dims},
        ]
        if loss is not None:
            data.append({"MetricName": "Loss", "Value": float(loss),
                         "Unit": "None", "Dimensions": dims})

        try:
            self._client.put_metric_data(Namespace=self.namespace,
                                         MetricData=data)
        except Exception as exc:
            # Metrics are best-effort; never kill a training job over them.
            print(f"[cloudwatch] publish failed (episode {episode}): {exc}")

    def publish_eval(self, episode, mean_reward, median_reward, max_reward,
                     furthest_x, flags, eval_episodes, score) -> None:
        """
        Publish near-greedy evaluation results (the best-model signal).

        Training-episode reward is noisy while epsilon is high; these
        eval metrics are what actually tracks progress toward reliable
        flag completion. Eval outcomes on this level are BIMODAL, so
        the MEDIAN (not the mean) plus the FLAG RATE are the honest
        numbers: EvalScore is the exact quantity that gates the
        best-checkpoint save (flag_rate * 100000 + median), so the
        dashboard shows exactly when and why the best checkpoint
        advanced. EvalFlagRate approaching 1.0 is the "level solved"
        indicator. MeanEvalReward / MaxEvalReward / EvalFurthestX /
        EvalFlagRuns keep their original names so existing dashboards
        continue to plot.
        """
        if not self.enabled:
            return

        dims = [{"Name": "RunId", "Value": self.run_id}]
        data = [
            {"MetricName": "MeanEvalReward", "Value": float(mean_reward),
             "Unit": "None", "Dimensions": dims},
            {"MetricName": "MedianEvalReward", "Value": float(median_reward),
             "Unit": "None", "Dimensions": dims},
            {"MetricName": "MaxEvalReward", "Value": float(max_reward),
             "Unit": "None", "Dimensions": dims},
            {"MetricName": "EvalFurthestX", "Value": float(furthest_x),
             "Unit": "None", "Dimensions": dims},
            {"MetricName": "EvalFlagRuns", "Value": float(flags),
             "Unit": "Count", "Dimensions": dims},
            {"MetricName": "EvalFlagRate",
             "Value": float(flags) / float(eval_episodes or 1),
             "Unit": "None", "Dimensions": dims},
            {"MetricName": "EvalScore", "Value": float(score),
             "Unit": "None", "Dimensions": dims},
        ]
        try:
            self._client.put_metric_data(Namespace=self.namespace,
                                         MetricData=data)
        except Exception as exc:
            print(f"[cloudwatch] eval publish failed (episode {episode}): {exc}")
