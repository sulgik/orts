// Runs GrowthBook's own contextual bandit engine on JSON lines from stdin.
import * as readline from "readline";
import { computeContextualBanditWeights } from "stats-ts/src/contextualBanditWeights";

const rl = readline.createInterface({ input: process.stdin });
rl.on("line", (line) => {
  const input = JSON.parse(line);
  const out = computeContextualBanditWeights(input);
  process.stdout.write(
    JSON.stringify({
      responses: out.responses.map((r) => ({
        context: r.context,
        leafId: r.leafId,
        weights: r.updatedWeights,
      })),
    }) + "\n",
  );
});
