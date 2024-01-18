using System;
using System.Collections.Generic;

namespace CopeLab.Library.algo.zoom
{
    public class IntInt
    {
        public readonly int left, right;

        public IntInt(int left, int right)
        {
            this.left = left;
            this.right = right;
        }
    }

    public class DoubleInt
    {
        public readonly double left;
        public readonly int right;

        public DoubleInt(double left, int right)
        {
            this.left = left;
            this.right = right;
        }
    }

    public class OOM
    {
        public const int ActionSample = 0;
        public const int ActionZoom = 1;

        private readonly double _muStar, _minInterval, _maxInterval, _coefCi;
        private readonly int _T, _K;
        private readonly int[][] _armDict;
        private int _lastIndex;
        public readonly Dictionary<int, OOM> children = new();

        public OOM(double muStar, double minInterval, double maxInterval, int T, int K, double coefCi = 1.5)
        {
            _muStar = muStar;
            _minInterval = minInterval;
            _maxInterval = maxInterval;
            _T = T;
            _K = K;
            _coefCi = coefCi;

            // the dictionnary that stores for each arm k of the grid : Number of positive observation, number of total sample, relation according to (7), relation according to (8)
            _armDict = new int[K + 1][];
            _armDict[0] = new[] { 0, 0, -2, -2 }; //extremities never need to be sampled
            for (var i = 1; i < K; i++)
                _armDict[i] = new[] { 0, 0, 0, 0 };
            _armDict[K] = new[] { 0, 0, 2, 2 };

            _lastIndex = K / 2; //begins at the middle of the grid
        }


        public double ReverseConverter(double k)
        {
            // function to map the grid index k to its input value s_d,n,k
            return k * (_maxInterval - _minInterval) / _K + _minInterval;
        }

        private IntInt ComputePointRelation(int k)
        {
            var totalObs = _armDict[k][1];

            if (k == 0 || k >= _K || totalObs == 0)
                throw new Exception("OOM.computePointRelation - invalid value: k=" + k + ", totalPulls=" +
                                    totalObs);

            var estimator = _armDict[k][0] / (double)totalObs;

            var b = Math.Sqrt(_coefCi * Math.Log(_T) / totalObs);
            var klError = Math.Log(_T / (double)totalObs) / totalObs;

            var relationCi = estimator + b < _muStar
                ? -2
                : (estimator - b > _muStar ? 2 : (estimator > _muStar ? 1 : -1));

            var d = ComputeKlDist(estimator, _muStar);
            var relationKl = estimator > _muStar ? (d >= klError ? 2 : 1) : (d >= klError ? -2 : -1);

            return new IntInt(relationCi, relationKl);
        }

        private static double ComputeKlDist(double hmu, double mu)
        {
            return mu == hmu
                ? 0.0
                : (mu > hmu
                    ? (hmu > 0 ? KlLog(hmu, mu) : -Math.Log(1 - mu))
                    : (hmu < 1 ? KlLog(hmu, mu) : -Math.Log(mu)));
        }

        private static double KlLog(double hmu, double mu)
        {
            return hmu * Math.Log(hmu / mu) + (1 - hmu) * Math.Log((1 - hmu) / (1 - mu));
        }

        public IntInt WhatToDo(bool optimistic)
        {
            var action = ActionSample;
            var index = _lastIndex;

            var p0 = _armDict[_lastIndex][0];
            var t0 = _armDict[_lastIndex][1];

            if (t0 > 0)
            {
                var highIndex = _lastIndex + (p0 / (double)t0 > _muStar ? 0 : 1);
                var lowIndex = highIndex - 1;

                var r0 = _armDict[lowIndex][optimistic ? 3 : 2];
                var r1 = _armDict[highIndex][optimistic ? 3 : 2];
                var t1 = _armDict[highIndex][1];

                if (r0 == -2)
                {
                    action = r1 == 2 ? ActionZoom : ActionSample;
                    index = r1 == 2 ? lowIndex : highIndex;
                }
                else
                    index = r1 == 2 ? lowIndex : (t1 == 0 ? highIndex : (t0 < t1 ? lowIndex : highIndex));
            }

            return new IntInt(action, action == ActionSample ? _lastIndex = index : index);
        }

        public OOM Zoom(int intervalIndex)
        {
            var size = (_maxInterval - _minInterval) / _K;
            var min = _minInterval + intervalIndex * size;
            return new OOM(_muStar, min, min + size, _T, _K, _coefCi);
        }

        public void UpdateArm(bool answer)
        {
            var index = _lastIndex;
            _armDict[index][0] += answer ? 1 : 0;
            _armDict[index][1]++;
            var ci = ComputePointRelation(index);
            _armDict[index][2] = ci.left;
            _armDict[index][3] = ci.right;
            zoom.ZOOM.LOG("OOM.updateArm - " + (answer ? 1 : 0) + " index=" + index + " A[" + _armDict[index][0] +
                          ", " +
                          _armDict[index][1] + ", " + _armDict[index][2] + ", " + _armDict[index][3] + "]");
        }

        public DoubleInt GetMostPulledArm()
        {
            var maxValue = 0;
            var maxIndex = _lastIndex;
            for (var index = 0; index < _armDict.Length; index++)
                if (_armDict[index][1] > maxValue)
                    maxValue = _armDict[maxIndex = index][1];

            zoom.ZOOM.LOG("OOM.getMostPulledArm - max=" + maxValue + ", index=" + maxIndex + ", reverse=" +
                          ReverseConverter(maxIndex));
            return new DoubleInt(ReverseConverter(maxIndex), maxValue);
        }
    }

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
        private static bool DebugMode = false;

        private readonly OOM _treeRoot;
        private OOM _oom = null;
        private bool _pullOptimistic = false;

        public ZOOM(double muStar, int T, int K = 0, double minInterval = 0.0, double maxInterval = 1.0)
        {
            if (K == 0) // set automatically to sqrt T
                K = (int)(Math.Sqrt(T / (Math.Log(T) * Math.Log(Math.Log(T)))));

            if (K % 2 == 1)
                K--;

            if (K < 4) // 4 is a minimum value to avoid some edge case
                K = 4;

            Console.WriteLine("K=" + K);

            _treeRoot = new OOM(muStar, minInterval, maxInterval, T, K, 1.5);
        }

        public double ChooseArm()
        {
            _oom = _treeRoot;
            int index;
            while (true)
            {
                IntInt what = _oom.WhatToDo(_pullOptimistic);
                index = what.right;
                if (what.left == OOM.ActionSample)
                    break;
                else
                {
                    if (!_oom.children.ContainsKey(index))
                        _oom.children.Add(index, _oom.Zoom(index));
                    _oom = _oom.children[index];
                }
            }

            return _oom.ReverseConverter(index);
        }

        public void UpdateArm(bool answer)
        {
            _oom.UpdateArm(answer);
            _pullOptimistic = !_pullOptimistic;
        }

        public double ReturnArm()
        {
            _oom = _treeRoot;
            while (true)
            {
                var what = _oom.WhatToDo(true);
                if (what.left == OOM.ActionSample || !_oom.children.ContainsKey(what.right))
                    break;
                _oom = _oom.children[what.right];
            }
            return _oom.GetMostPulledArm().left;
        }

        public static void LOG(String msg)
        {
            if (DebugMode)
                Console.WriteLine(msg);
        }

        public static void Main(string[] args)
        {
            const int budget = 20;
            const double stimulusTarget = 0.3;

            var zoom = new ZOOM(0.5, budget, 32);
            for (var i = 0; i < budget; i++)
            {
                var value = zoom.ChooseArm();
                var answer = value > stimulusTarget;

                var doInvert = i % 7 == 0;
                
                if (doInvert) //nobody's perfect !
                    answer = !answer;

                Console.WriteLine((i + 1) + ") value=" + value + " " + answer + (doInvert ? "*" : ""));
                zoom.UpdateArm(answer);
            }

            var result = zoom.ReturnArm();
            Console.WriteLine("Result=" + result);
        }
    }
}