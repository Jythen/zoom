import java.util.Arrays;
import java.util.HashMap;

/**
 * Class that implement the ZOOM Method, using a tree of OOM
 * Parameters
 * mu_star :  the desired output (float)
 * mininterval : the minimun input value of the current grid. (float)
 * maxinterval : the maximum input value of the current grid. (float)
 * T : The total sampling budget
 * K : The coarseness of the grid
 */
public class ZOOM
{
    public static final boolean DEBUG_MODE = false;

    public static final int ACTION_SAMPLE = 0;
    public static final int ACTION_ZOOM = 1;

    public static class IntInt
    {
        public final int left, right;

        public IntInt(int left, int right)
        {
            this.left = left;
            this.right = right;
        }
    }

    public static class DoubleInt
    {
        public final double left;
        public final int right;

        public DoubleInt(double left, int right)
        {
            this.left = left;
            this.right = right;
        }
    }

    public static class OOM
    {
        private final double muStar, minInterval, maxInterval, coefCI;
        private final int T, K;
        private final int[][] armDict;
        private int lastIndex;
        HashMap<Integer, OOM> children = new HashMap<>();

        public OOM(double muStar, double minInterval, double maxInterval, int T, int K, double coefCI)
        {
            this.muStar = muStar;
            this.minInterval = minInterval;
            this.maxInterval = maxInterval;
            this.T = T;
            this.K = K;
            // the dictionnary that stores for each arm k of the grid : Number of positive observation, number of total sample, relation according to (7), relation according to (8)
            this.armDict = new int[K + 1][4];
            this.armDict[0] = new int[]{0, 0, -2, -2}; //extremities never need to be sampled
            this.armDict[K] = new int[]{0, 0, 2, 2};

            this.lastIndex = K / 2; //begins at the middle of the grid
            this.coefCI = coefCI;
        }

        public String armDictToString()
        {
            StringBuilder s = new StringBuilder();
            for (int i = 0; i < armDict.length; i++)
                s.append(i).append(": ").append(Arrays.toString(armDict[i])).append(", ");
            return s.toString();
        }

        public double reverseConverter(double k)
        {
            // function to map the grid index k to its input value s_d,n,k
            return k * (maxInterval - minInterval) / K + minInterval;
        }

        private IntInt computePointRelation(int k)
        {
            int totalObs = armDict[k][1];

            if (k == 0 || k >= K || totalObs == 0)
                throw new RuntimeException("OOM.computePointRelation - invalid value: k=" + k + ", totalPulls=" + totalObs);

            double estimator = armDict[k][0] / (double) totalObs;

            double b = Math.sqrt(coefCI * Math.log(T) / totalObs);
            double klError = Math.log(T / (double) totalObs) / totalObs;

            int relationCI = estimator + b < muStar ? -2 : (estimator - b > muStar ? 2 : (estimator > muStar ? 1 : -1));

            double d = computeKLDist(estimator, muStar);
            int relationKL = estimator > muStar ? (d >= klError ? 2 : 1) : (d >= klError ? -2 : -1);

            return new IntInt(relationCI, relationKL);
        }

        private static double computeKLDist(double hmu, double mu)
        {
            return mu == hmu ? 0.0 : (mu > hmu ? (hmu > 0 ? klLog(hmu, mu) : -Math.log(1 - mu)) : (hmu < 1 ? klLog(hmu, mu) : -Math.log(mu)));
        }

        private static double klLog(double hmu, double mu)
        {
            return hmu * Math.log(hmu / mu) + (1 - hmu) * Math.log((1 - hmu) / (1 - mu));
        }

        public IntInt whatToDo(boolean optimistic)
        {
            int action = ACTION_SAMPLE;
            int index = lastIndex;

            int p0 = armDict[lastIndex][0];
            int t0 = armDict[lastIndex][1];

            if (t0 > 0)
            {
                int highIndex = lastIndex + (p0 / (double) t0 > muStar ? 0 : 1);
                int lowIndex = highIndex - 1;

                int r0 = armDict[lowIndex][optimistic ? 3 : 2];
                int r1 = armDict[highIndex][optimistic ? 3 : 2];
                int t1 = armDict[highIndex][1];

                if (r0 == -2)
                {
                    action = r1 == 2 ? ACTION_ZOOM : ACTION_SAMPLE;
                    index = r1 == 2 ? lowIndex : highIndex;
                } else
                    index = r1 == 2 ? lowIndex : (t1 == 0 ? highIndex : (t0 < t1 ? lowIndex : highIndex));
            }

            return new IntInt(action, action == ACTION_SAMPLE ? lastIndex = index : index);
        }

        public OOM zoom(int intervalIndex)
        {
            double size = (maxInterval - minInterval) / K;
            double min = minInterval + intervalIndex * size;
            return new OOM(muStar, min, min + size, T, K, coefCI);
        }

        public void updateArm(boolean answer)
        {
            int index = lastIndex;
            armDict[index][0] += answer ? 1 : 0;
            armDict[index][1]++;
            IntInt ci = computePointRelation(index);
            armDict[index][2] = ci.left;
            armDict[index][3] = ci.right;
            ZOOM.log("OOM.updateArm - " + (answer ? 1 : 0) + " index=" + index + " A[" + armDict[index][0] + ", " + armDict[index][1] + ", " + armDict[index][2] + ", " + armDict[index][3] + "]");
        }

        public DoubleInt getMostPulledArm()
        {
            int maxValue = 0;
            int maxIndex = lastIndex;
            for (int index = 0; index < armDict.length; index++)
                if (armDict[index][1] > maxValue)
                    maxValue = armDict[maxIndex = index][1];

            ZOOM.log("OOM.getMostPulledArm - max=" + maxValue + ", index=" + maxIndex + ", reverse=" + reverseConverter(maxIndex));
            return new DoubleInt(reverseConverter(maxIndex), maxValue);
        }
    }

    private final OOM treeRoot;
    private OOM oom = null;
    private boolean pullOptimistic = false;

    public ZOOM(double muStar, int T, int K)
    {
        this(muStar, T, K, 0.0, 1.0);
    }

    public ZOOM(double muStar, int T, int K, double minInterval, double maxInterval)
    {
        if (K == 0)  // set automatically to sqrt T
            K = (int) (Math.sqrt(T / (Math.log(T) * Math.log(Math.log(T)))));

        System.out.println("K=" + K);

        if (K % 2 == 1)
            K--;

        if (K < 4) // 4 is a minimum value to avoid some edge case
            K = 4;

        this.treeRoot = new OOM(muStar, minInterval, maxInterval, T, K, 1.5);
    }

    public double chooseArm()
    {
        oom = treeRoot;
        int index;
        while (true)
        {
            IntInt what = oom.whatToDo(pullOptimistic);
            index = what.right;
            if (what.left == ACTION_SAMPLE)
                break;
            else
            {
                if (!oom.children.containsKey(index))
                    oom.children.put(index, oom.zoom(index));
                oom = oom.children.get(index);
            }
        }
        return oom.reverseConverter(index);
    }

    public void updateArm(boolean answer)
    {
        oom.updateArm(answer);
        pullOptimistic = !pullOptimistic;
    }

    public double returnArm()
    {
        oom = treeRoot;
        while (true)
        {
            IntInt what = oom.whatToDo(true);
            if (what.left == ACTION_SAMPLE || !oom.children.containsKey(what.right))
                break;
            oom = oom.children.get(what.right);
        }
        return oom.getMostPulledArm().left;
    }

    public static void log(String msg)
    {
        if (DEBUG_MODE)
            System.out.println(msg);
    }

    public static void main(String... args)
    {
        int budget = 50;

        ZOOM zoom = new ZOOM(0.5, budget, 0);
        for (int i = 0; i < budget; i++)
        {
            double value = zoom.chooseArm();
            boolean answer = value > 0.65;

            boolean doInvert = i % 7 == 0;

            if (doInvert) //nobody's perfect !
                answer = !answer;

            System.out.println((i + 1) + ") value=" + value + " " + answer + (doInvert ? "*" : ""));
            zoom.updateArm(answer);
        }

        double result = zoom.returnArm();
        System.out.println("Result=" + result);
    }

}
