\ENCODING=ISO-8859-1
\Problem name: 

Minimize
 obj: wcvar
Subject To
 w_0_ADM:                        w_0_ADM - 0.99 b_0_ADM + 1.01 s_0_ADM  = 0.1
 w_0_AXP:                        w_0_AXP - 0.99 b_0_AXP + 1.01 s_0_AXP  = 0.1
 w_0_DIS:                        w_0_DIS - 0.99 b_0_DIS + 1.01 s_0_DIS  = 0.1
 w_0_FDX:                        w_0_FDX - 0.99 b_0_FDX + 1.01 s_0_FDX  = 0.1
 w_0_IBM:                        w_0_IBM - 0.99 b_0_IBM + 1.01 s_0_IBM  = 0.1
 w_0_INTC:                       w_0_INTC - 0.99 b_0_INTC + 1.01 s_0_INTC
                                  = 0.1
 w_0_KO:                         w_0_KO - 0.99 b_0_KO + 1.01 s_0_KO  = 0.1
 w_0_MAT:                        w_0_MAT - 0.99 b_0_MAT + 1.01 s_0_MAT  = 0.1
 w_0_MSFT:                       w_0_MSFT - 0.99 b_0_MSFT + 1.01 s_0_MSFT
                                  = 0.1
 w_0_PG:                         w_0_PG - 0.99 b_0_PG + 1.01 s_0_PG  = 0.1
 w_0_balance_constraint:         b_0_ADM + b_0_AXP + b_0_DIS + b_0_FDX
                                 + b_0_IBM + b_0_INTC + b_0_KO + b_0_MAT
                                 + b_0_MSFT + b_0_PG - s_0_ADM - s_0_AXP
                                 - s_0_DIS - s_0_FDX - s_0_IBM - s_0_INTC
                                 - s_0_KO - s_0_MAT - s_0_MSFT - s_0_PG  = 0
 w_k1_ADM_s0_return_constraint:  1.13158 w_0_ADM - w_k1_ADM_s0
                                 + 0.99 b_k1_ADM_s0 - 1.01 s_k1_ADM_s0  = 0
 w_k1_ADM_s1_return_constraint:  0.930952 w_0_ADM - w_k1_ADM_s1
                                 + 0.99 b_k1_ADM_s1 - 1.01 s_k1_ADM_s1  = 0
 w_k1_ADM_s00_return_constraint: 1.01209 w_k1_ADM_s0 - w_k1_ADM_s00
                                 + 0.99 b_k1_ADM_s00 - 1.01 s_k1_ADM_s00  = 0
 w_k1_ADM_s01_return_constraint: 0.95882 w_k1_ADM_s0 - w_k1_ADM_s01
                                 + 0.99 b_k1_ADM_s01 - 1.01 s_k1_ADM_s01  = 0
 w_k1_ADM_s10_return_constraint: 1.03843 w_k1_ADM_s1 - w_k1_ADM_s10
                                 + 0.99 b_k1_ADM_s10 - 1.01 s_k1_ADM_s10  = 0
 w_k1_ADM_s11_return_constraint: 0.875039 w_k1_ADM_s1 - w_k1_ADM_s11
                                 + 0.99 b_k1_ADM_s11 - 1.01 s_k1_ADM_s11  = 0
 w_k1_ADM_s000_return_constraint: 0.79597 w_k1_ADM_s00 - w_k1_ADM_s000
                                 + 0.99 b_k1_ADM_s000 - 1.01 s_k1_ADM_s000  = 0
 w_k1_ADM_s001_return_constraint: 1.20968 w_k1_ADM_s00 - w_k1_ADM_s001
                                 + 0.99 b_k1_ADM_s001 - 1.01 s_k1_ADM_s001  = 0
 w_k1_ADM_s002_return_constraint: 0.938712 w_k1_ADM_s00 - w_k1_ADM_s002
                                 + 0.99 b_k1_ADM_s002 - 1.01 s_k1_ADM_s002  = 0
 w_k1_ADM_s003_return_constraint: 1.00253 w_k1_ADM_s00 - w_k1_ADM_s003
                                 + 0.99 b_k1_ADM_s003 - 1.01 s_k1_ADM_s003  = 0
 w_k1_ADM_s004_return_constraint: 1.15874 w_k1_ADM_s00 - w_k1_ADM_s004
                                 + 0.99 b_k1_ADM_s004 - 1.01 s_k1_ADM_s004  = 0
 w_k1_ADM_s005_return_constraint: 1.73457 w_k1_ADM_s00 - w_k1_ADM_s005
                                 + 0.99 b_k1_ADM_s005 - 1.01 s_k1_ADM_s005  = 0
 w_k1_ADM_s006_return_constraint: 0.774001 w_k1_ADM_s00 - w_k1_ADM_s006
                                 + 0.99 b_k1_ADM_s006 - 1.01 s_k1_ADM_s006  = 0
 w_k1_ADM_s007_return_constraint: 1.09298 w_k1_ADM_s00 - w_k1_ADM_s007
                                 + 0.99 b_k1_ADM_s007 - 1.01 s_k1_ADM_s007  = 0
 w_k1_ADM_s010_return_constraint: 0.733914 w_k1_ADM_s01 - w_k1_ADM_s010
                                 + 0.99 b_k1_ADM_s010 - 1.01 s_k1_ADM_s010  = 0
 w_k1_ADM_s011_return_constraint: 0.654849 w_k1_ADM_s01 - w_k1_ADM_s011
                                 + 0.99 b_k1_ADM_s011 - 1.01 s_k1_ADM_s011  = 0
 w_k1_ADM_s012_return_constraint: 1.1915 w_k1_ADM_s01 - w_k1_ADM_s012
                                 + 0.99 b_k1_ADM_s012 - 1.01 s_k1_ADM_s012  = 0
 w_k1_ADM_s013_return_constraint: 0.950088 w_k1_ADM_s01 - w_k1_ADM_s013
                                 + 0.99 b_k1_ADM_s013 - 1.01 s_k1_ADM_s013  = 0
 w_k1_ADM_s014_return_constraint: 0.886581 w_k1_ADM_s01 - w_k1_ADM_s014
                                 + 0.99 b_k1_ADM_s014 - 1.01 s_k1_ADM_s014  = 0
 w_k1_ADM_s015_return_constraint: 0.942061 w_k1_ADM_s01 - w_k1_ADM_s015
                                 + 0.99 b_k1_ADM_s015 - 1.01 s_k1_ADM_s015  = 0
 w_k1_ADM_s016_return_constraint: 1.2178 w_k1_ADM_s01 - w_k1_ADM_s016
                                 + 0.99 b_k1_ADM_s016 - 1.01 s_k1_ADM_s016  = 0
 w_k1_ADM_s017_return_constraint: 1.46036 w_k1_ADM_s01 - w_k1_ADM_s017
                                 + 0.99 b_k1_ADM_s017 - 1.01 s_k1_ADM_s017  = 0
 w_k1_ADM_s100_return_constraint: 1.0002 w_k1_ADM_s10 - w_k1_ADM_s100
                                 + 0.99 b_k1_ADM_s100 - 1.01 s_k1_ADM_s100  = 0
 w_k1_ADM_s101_return_constraint: 1.05845 w_k1_ADM_s10 - w_k1_ADM_s101
                                 + 0.99 b_k1_ADM_s101 - 1.01 s_k1_ADM_s101  = 0
 w_k1_ADM_s102_return_constraint: 0.996108 w_k1_ADM_s10 - w_k1_ADM_s102
                                 + 0.99 b_k1_ADM_s102 - 1.01 s_k1_ADM_s102  = 0
 w_k1_ADM_s103_return_constraint: 1.22402 w_k1_ADM_s10 - w_k1_ADM_s103
                                 + 0.99 b_k1_ADM_s103 - 1.01 s_k1_ADM_s103  = 0
 w_k1_ADM_s104_return_constraint: 1.36397 w_k1_ADM_s10 - w_k1_ADM_s104
                                 + 0.99 b_k1_ADM_s104 - 1.01 s_k1_ADM_s104  = 0
 w_k1_ADM_s105_return_constraint: 0.982084 w_k1_ADM_s10 - w_k1_ADM_s105
                                 + 0.99 b_k1_ADM_s105 - 1.01 s_k1_ADM_s105  = 0
 w_k1_ADM_s106_return_constraint: 1.14179 w_k1_ADM_s10 - w_k1_ADM_s106
                                 + 0.99 b_k1_ADM_s106 - 1.01 s_k1_ADM_s106  = 0
 w_k1_ADM_s107_return_constraint: 0.858518 w_k1_ADM_s10 - w_k1_ADM_s107
                                 + 0.99 b_k1_ADM_s107 - 1.01 s_k1_ADM_s107  = 0
 w_k1_ADM_s110_return_constraint: 0.832668 w_k1_ADM_s11 - w_k1_ADM_s110
                                 + 0.99 b_k1_ADM_s110 - 1.01 s_k1_ADM_s110  = 0
 w_k1_ADM_s111_return_constraint: 0.886175 w_k1_ADM_s11 - w_k1_ADM_s111
                                 + 0.99 b_k1_ADM_s111 - 1.01 s_k1_ADM_s111  = 0
 w_k1_ADM_s112_return_constraint: 1.45598 w_k1_ADM_s11 - w_k1_ADM_s112
                                 + 0.99 b_k1_ADM_s112 - 1.01 s_k1_ADM_s112  = 0
 w_k1_ADM_s113_return_constraint: 1.20882 w_k1_ADM_s11 - w_k1_ADM_s113
                                 + 0.99 b_k1_ADM_s113 - 1.01 s_k1_ADM_s113  = 0
 w_k1_ADM_s114_return_constraint: 1.10881 w_k1_ADM_s11 - w_k1_ADM_s114
                                 + 0.99 b_k1_ADM_s114 - 1.01 s_k1_ADM_s114  = 0
 w_k1_ADM_s115_return_constraint: 0.755767 w_k1_ADM_s11 - w_k1_ADM_s115
                                 + 0.99 b_k1_ADM_s115 - 1.01 s_k1_ADM_s115  = 0
 w_k1_ADM_s116_return_constraint: 0.86793 w_k1_ADM_s11 - w_k1_ADM_s116
                                 + 0.99 b_k1_ADM_s116 - 1.01 s_k1_ADM_s116  = 0
 w_k1_ADM_s117_return_constraint: 0.923907 w_k1_ADM_s11 - w_k1_ADM_s117
                                 + 0.99 b_k1_ADM_s117 - 1.01 s_k1_ADM_s117  = 0
 w_k1_AXP_s0_return_constraint:  1.16878 w_0_AXP - w_k1_AXP_s0
                                 + 0.99 b_k1_AXP_s0 - 1.01 s_k1_AXP_s0  = 0
 w_k1_AXP_s1_return_constraint:  0.83794 w_0_AXP - w_k1_AXP_s1
                                 + 0.99 b_k1_AXP_s1 - 1.01 s_k1_AXP_s1  = 0
 w_k1_AXP_s00_return_constraint: 1.03951 w_k1_AXP_s0 - w_k1_AXP_s00
                                 + 0.99 b_k1_AXP_s00 - 1.01 s_k1_AXP_s00  = 0
 w_k1_AXP_s01_return_constraint: 1.14156 w_k1_AXP_s0 - w_k1_AXP_s01
                                 + 0.99 b_k1_AXP_s01 - 1.01 s_k1_AXP_s01  = 0
 w_k1_AXP_s10_return_constraint: 1.22037 w_k1_AXP_s1 - w_k1_AXP_s10
                                 + 0.99 b_k1_AXP_s10 - 1.01 s_k1_AXP_s10  = 0
 w_k1_AXP_s11_return_constraint: 0.870374 w_k1_AXP_s1 - w_k1_AXP_s11
                                 + 0.99 b_k1_AXP_s11 - 1.01 s_k1_AXP_s11  = 0
 w_k1_AXP_s000_return_constraint: 0.683881 w_k1_AXP_s00 - w_k1_AXP_s000
                                 + 0.99 b_k1_AXP_s000 - 1.01 s_k1_AXP_s000  = 0
 w_k1_AXP_s001_return_constraint: 1.45674 w_k1_AXP_s00 - w_k1_AXP_s001
                                 + 0.99 b_k1_AXP_s001 - 1.01 s_k1_AXP_s001  = 0
 w_k1_AXP_s002_return_constraint: 0.876576 w_k1_AXP_s00 - w_k1_AXP_s002
                                 + 0.99 b_k1_AXP_s002 - 1.01 s_k1_AXP_s002  = 0
 w_k1_AXP_s003_return_constraint: 1.01148 w_k1_AXP_s00 - w_k1_AXP_s003
                                 + 0.99 b_k1_AXP_s003 - 1.01 s_k1_AXP_s003  = 0
 w_k1_AXP_s004_return_constraint: 1.01684 w_k1_AXP_s00 - w_k1_AXP_s004
                                 + 0.99 b_k1_AXP_s004 - 1.01 s_k1_AXP_s004  = 0
 w_k1_AXP_s005_return_constraint: 1.82464 w_k1_AXP_s00 - w_k1_AXP_s005
                                 + 0.99 b_k1_AXP_s005 - 1.01 s_k1_AXP_s005  = 0
 w_k1_AXP_s006_return_constraint: 0.627121 w_k1_AXP_s00 - w_k1_AXP_s006
                                 + 0.99 b_k1_AXP_s006 - 1.01 s_k1_AXP_s006  = 0
 w_k1_AXP_s007_return_constraint: 1.13526 w_k1_AXP_s00 - w_k1_AXP_s007
                                 + 0.99 b_k1_AXP_s007 - 1.01 s_k1_AXP_s007  = 0
 w_k1_AXP_s010_return_constraint: 0.620139 w_k1_AXP_s01 - w_k1_AXP_s010
                                 + 0.99 b_k1_AXP_s010 - 1.01 s_k1_AXP_s010  = 0
 w_k1_AXP_s011_return_constraint: 0.589133 w_k1_AXP_s01 - w_k1_AXP_s011
                                 + 0.99 b_k1_AXP_s011 - 1.01 s_k1_AXP_s011  = 0
 w_k1_AXP_s012_return_constraint: 1.06029 w_k1_AXP_s01 - w_k1_AXP_s012
                                 + 0.99 b_k1_AXP_s012 - 1.01 s_k1_AXP_s012  = 0
 w_k1_AXP_s013_return_constraint: 1.11404 w_k1_AXP_s01 - w_k1_AXP_s013
                                 + 0.99 b_k1_AXP_s013 - 1.01 s_k1_AXP_s013  = 0
 w_k1_AXP_s014_return_constraint: 0.820552 w_k1_AXP_s01 - w_k1_AXP_s014
                                 + 0.99 b_k1_AXP_s014 - 1.01 s_k1_AXP_s014  = 0
 w_k1_AXP_s015_return_constraint: 0.877839 w_k1_AXP_s01 - w_k1_AXP_s015
                                 + 0.99 b_k1_AXP_s015 - 1.01 s_k1_AXP_s015  = 0
 w_k1_AXP_s016_return_constraint: 1.63328 w_k1_AXP_s01 - w_k1_AXP_s016
                                 + 0.99 b_k1_AXP_s016 - 1.01 s_k1_AXP_s016  = 0
 w_k1_AXP_s017_return_constraint: 1.43113 w_k1_AXP_s01 - w_k1_AXP_s017
                                 + 0.99 b_k1_AXP_s017 - 1.01 s_k1_AXP_s017  = 0
 w_k1_AXP_s100_return_constraint: 0.918147 w_k1_AXP_s10 - w_k1_AXP_s100
                                 + 0.99 b_k1_AXP_s100 - 1.01 s_k1_AXP_s100  = 0
 w_k1_AXP_s101_return_constraint: 0.936556 w_k1_AXP_s10 - w_k1_AXP_s101
                                 + 0.99 b_k1_AXP_s101 - 1.01 s_k1_AXP_s101  = 0
 w_k1_AXP_s102_return_constraint: 0.546699 w_k1_AXP_s10 - w_k1_AXP_s102
                                 + 0.99 b_k1_AXP_s102 - 1.01 s_k1_AXP_s102  = 0
 w_k1_AXP_s103_return_constraint: 1.38919 w_k1_AXP_s10 - w_k1_AXP_s103
                                 + 0.99 b_k1_AXP_s103 - 1.01 s_k1_AXP_s103  = 0
 w_k1_AXP_s104_return_constraint: 1.37088 w_k1_AXP_s10 - w_k1_AXP_s104
                                 + 0.99 b_k1_AXP_s104 - 1.01 s_k1_AXP_s104  = 0
 w_k1_AXP_s105_return_constraint: 0.944338 w_k1_AXP_s10 - w_k1_AXP_s105
                                 + 0.99 b_k1_AXP_s105 - 1.01 s_k1_AXP_s105  = 0
 w_k1_AXP_s106_return_constraint: 0.959994 w_k1_AXP_s10 - w_k1_AXP_s106
                                 + 0.99 b_k1_AXP_s106 - 1.01 s_k1_AXP_s106  = 0
 w_k1_AXP_s107_return_constraint: 0.82977 w_k1_AXP_s10 - w_k1_AXP_s107
                                 + 0.99 b_k1_AXP_s107 - 1.01 s_k1_AXP_s107  = 0
 w_k1_AXP_s110_return_constraint: 0.676033 w_k1_AXP_s11 - w_k1_AXP_s110
                                 + 0.99 b_k1_AXP_s110 - 1.01 s_k1_AXP_s110  = 0
 w_k1_AXP_s111_return_constraint: 0.751013 w_k1_AXP_s11 - w_k1_AXP_s111
                                 + 0.99 b_k1_AXP_s111 - 1.01 s_k1_AXP_s111  = 0
 w_k1_AXP_s112_return_constraint: 1.435 w_k1_AXP_s11 - w_k1_AXP_s112
                                 + 0.99 b_k1_AXP_s112 - 1.01 s_k1_AXP_s112  = 0
 w_k1_AXP_s113_return_constraint: 1.04316 w_k1_AXP_s11 - w_k1_AXP_s113
                                 + 0.99 b_k1_AXP_s113 - 1.01 s_k1_AXP_s113  = 0
 w_k1_AXP_s114_return_constraint: 1.1919 w_k1_AXP_s11 - w_k1_AXP_s114
                                 + 0.99 b_k1_AXP_s114 - 1.01 s_k1_AXP_s114  = 0
 w_k1_AXP_s115_return_constraint: 0.772012 w_k1_AXP_s11 - w_k1_AXP_s115
                                 + 0.99 b_k1_AXP_s115 - 1.01 s_k1_AXP_s115  = 0
 w_k1_AXP_s116_return_constraint: 0.721488 w_k1_AXP_s11 - w_k1_AXP_s116
                                 + 0.99 b_k1_AXP_s116 - 1.01 s_k1_AXP_s116  = 0
 w_k1_AXP_s117_return_constraint: 1.08065 w_k1_AXP_s11 - w_k1_AXP_s117
                                 + 0.99 b_k1_AXP_s117 - 1.01 s_k1_AXP_s117  = 0
 w_k1_DIS_s0_return_constraint:  1.19655 w_0_DIS - w_k1_DIS_s0
                                 + 0.99 b_k1_DIS_s0 - 1.01 s_k1_DIS_s0  = 0
 w_k1_DIS_s1_return_constraint:  0.822456 w_0_DIS - w_k1_DIS_s1
                                 + 0.99 b_k1_DIS_s1 - 1.01 s_k1_DIS_s1  = 0
 w_k1_DIS_s00_return_constraint: 1.0583 w_k1_DIS_s0 - w_k1_DIS_s00
                                 + 0.99 b_k1_DIS_s00 - 1.01 s_k1_DIS_s00  = 0
 w_k1_DIS_s01_return_constraint: 1.04171 w_k1_DIS_s0 - w_k1_DIS_s01
                                 + 0.99 b_k1_DIS_s01 - 1.01 s_k1_DIS_s01  = 0
 w_k1_DIS_s10_return_constraint: 1.27257 w_k1_DIS_s1 - w_k1_DIS_s10
                                 + 0.99 b_k1_DIS_s10 - 1.01 s_k1_DIS_s10  = 0
 w_k1_DIS_s11_return_constraint: 0.776708 w_k1_DIS_s1 - w_k1_DIS_s11
                                 + 0.99 b_k1_DIS_s11 - 1.01 s_k1_DIS_s11  = 0
 w_k1_DIS_s000_return_constraint: 0.679431 w_k1_DIS_s00 - w_k1_DIS_s000
                                 + 0.99 b_k1_DIS_s000 - 1.01 s_k1_DIS_s000  = 0
 w_k1_DIS_s001_return_constraint: 1.27699 w_k1_DIS_s00 - w_k1_DIS_s001
                                 + 0.99 b_k1_DIS_s001 - 1.01 s_k1_DIS_s001  = 0
 w_k1_DIS_s002_return_constraint: 0.877428 w_k1_DIS_s00 - w_k1_DIS_s002
                                 + 0.99 b_k1_DIS_s002 - 1.01 s_k1_DIS_s002  = 0
 w_k1_DIS_s003_return_constraint: 0.814351 w_k1_DIS_s00 - w_k1_DIS_s003
                                 + 0.99 b_k1_DIS_s003 - 1.01 s_k1_DIS_s003  = 0
 w_k1_DIS_s004_return_constraint: 1.34938 w_k1_DIS_s00 - w_k1_DIS_s004
                                 + 0.99 b_k1_DIS_s004 - 1.01 s_k1_DIS_s004  = 0
 w_k1_DIS_s005_return_constraint: 2.19013 w_k1_DIS_s00 - w_k1_DIS_s005
                                 + 0.99 b_k1_DIS_s005 - 1.01 s_k1_DIS_s005  = 0
 w_k1_DIS_s006_return_constraint: 0.625855 w_k1_DIS_s00 - w_k1_DIS_s006
                                 + 0.99 b_k1_DIS_s006 - 1.01 s_k1_DIS_s006  = 0
 w_k1_DIS_s007_return_constraint: 1.26094 w_k1_DIS_s00 - w_k1_DIS_s007
                                 + 0.99 b_k1_DIS_s007 - 1.01 s_k1_DIS_s007  = 0
 w_k1_DIS_s010_return_constraint: 0.615788 w_k1_DIS_s01 - w_k1_DIS_s010
                                 + 0.99 b_k1_DIS_s010 - 1.01 s_k1_DIS_s010  = 0
 w_k1_DIS_s011_return_constraint: 0.583194 w_k1_DIS_s01 - w_k1_DIS_s011
                                 + 0.99 b_k1_DIS_s011 - 1.01 s_k1_DIS_s011  = 0
 w_k1_DIS_s012_return_constraint: 1.28533 w_k1_DIS_s01 - w_k1_DIS_s012
                                 + 0.99 b_k1_DIS_s012 - 1.01 s_k1_DIS_s012  = 0
 w_k1_DIS_s013_return_constraint: 1.06145 w_k1_DIS_s01 - w_k1_DIS_s013
                                 + 0.99 b_k1_DIS_s013 - 1.01 s_k1_DIS_s013  = 0
 w_k1_DIS_s014_return_constraint: 0.801737 w_k1_DIS_s01 - w_k1_DIS_s014
                                 + 0.99 b_k1_DIS_s014 - 1.01 s_k1_DIS_s014  = 0
 w_k1_DIS_s015_return_constraint: 1.01159 w_k1_DIS_s01 - w_k1_DIS_s015
                                 + 0.99 b_k1_DIS_s015 - 1.01 s_k1_DIS_s015  = 0
 w_k1_DIS_s016_return_constraint: 2.34318 w_k1_DIS_s01 - w_k1_DIS_s016
                                 + 0.99 b_k1_DIS_s016 - 1.01 s_k1_DIS_s016  = 0
 w_k1_DIS_s017_return_constraint: 1.58415 w_k1_DIS_s01 - w_k1_DIS_s017
                                 + 0.99 b_k1_DIS_s017 - 1.01 s_k1_DIS_s017  = 0
 w_k1_DIS_s100_return_constraint: 0.827886 w_k1_DIS_s10 - w_k1_DIS_s100
                                 + 0.99 b_k1_DIS_s100 - 1.01 s_k1_DIS_s100  = 0
 w_k1_DIS_s101_return_constraint: 0.752541 w_k1_DIS_s10 - w_k1_DIS_s101
                                 + 0.99 b_k1_DIS_s101 - 1.01 s_k1_DIS_s101  = 0
 w_k1_DIS_s102_return_constraint: 0.718141 w_k1_DIS_s10 - w_k1_DIS_s102
                                 + 0.99 b_k1_DIS_s102 - 1.01 s_k1_DIS_s102  = 0
 w_k1_DIS_s103_return_constraint: 1.36382 w_k1_DIS_s10 - w_k1_DIS_s103
                                 + 0.99 b_k1_DIS_s103 - 1.01 s_k1_DIS_s103  = 0
 w_k1_DIS_s104_return_constraint: 1.44174 w_k1_DIS_s10 - w_k1_DIS_s104
                                 + 0.99 b_k1_DIS_s104 - 1.01 s_k1_DIS_s104  = 0
 w_k1_DIS_s105_return_constraint: 0.870423 w_k1_DIS_s10 - w_k1_DIS_s105
                                 + 0.99 b_k1_DIS_s105 - 1.01 s_k1_DIS_s105  = 0
 w_k1_DIS_s106_return_constraint: 1.12579 w_k1_DIS_s10 - w_k1_DIS_s106
                                 + 0.99 b_k1_DIS_s106 - 1.01 s_k1_DIS_s106  = 0
 w_k1_DIS_s107_return_constraint: 0.693714 w_k1_DIS_s10 - w_k1_DIS_s107
                                 + 0.99 b_k1_DIS_s107 - 1.01 s_k1_DIS_s107  = 0
 w_k1_DIS_s110_return_constraint: 0.748282 w_k1_DIS_s11 - w_k1_DIS_s110
                                 + 0.99 b_k1_DIS_s110 - 1.01 s_k1_DIS_s110  = 0
 w_k1_DIS_s111_return_constraint: 0.830122 w_k1_DIS_s11 - w_k1_DIS_s111
                                 + 0.99 b_k1_DIS_s111 - 1.01 s_k1_DIS_s111  = 0
 w_k1_DIS_s112_return_constraint: 1.62753 w_k1_DIS_s11 - w_k1_DIS_s112
                                 + 0.99 b_k1_DIS_s112 - 1.01 s_k1_DIS_s112  = 0
 w_k1_DIS_s113_return_constraint: 1.23361 w_k1_DIS_s11 - w_k1_DIS_s113
                                 + 0.99 b_k1_DIS_s113 - 1.01 s_k1_DIS_s113  = 0
 w_k1_DIS_s114_return_constraint: 1.1978 w_k1_DIS_s11 - w_k1_DIS_s114
                                 + 0.99 b_k1_DIS_s114 - 1.01 s_k1_DIS_s114  = 0
 w_k1_DIS_s115_return_constraint: 0.612409 w_k1_DIS_s11 - w_k1_DIS_s115
                                 + 0.99 b_k1_DIS_s115 - 1.01 s_k1_DIS_s115  = 0
 w_k1_DIS_s116_return_constraint: 0.840878 w_k1_DIS_s11 - w_k1_DIS_s116
                                 + 0.99 b_k1_DIS_s116 - 1.01 s_k1_DIS_s116  = 0
 w_k1_DIS_s117_return_constraint: 0.939242 w_k1_DIS_s11 - w_k1_DIS_s117
                                 + 0.99 b_k1_DIS_s117 - 1.01 s_k1_DIS_s117  = 0
 w_k1_FDX_s0_return_constraint:  1.13565 w_0_FDX - w_k1_FDX_s0
                                 + 0.99 b_k1_FDX_s0 - 1.01 s_k1_FDX_s0  = 0
 w_k1_FDX_s1_return_constraint:  0.891862 w_0_FDX - w_k1_FDX_s1
                                 + 0.99 b_k1_FDX_s1 - 1.01 s_k1_FDX_s1  = 0
 w_k1_FDX_s00_return_constraint: 1.03065 w_k1_FDX_s0 - w_k1_FDX_s00
                                 + 0.99 b_k1_FDX_s00 - 1.01 s_k1_FDX_s00  = 0
 w_k1_FDX_s01_return_constraint: 0.970237 w_k1_FDX_s0 - w_k1_FDX_s01
                                 + 0.99 b_k1_FDX_s01 - 1.01 s_k1_FDX_s01  = 0
 w_k1_FDX_s10_return_constraint: 1.17521 w_k1_FDX_s1 - w_k1_FDX_s10
                                 + 0.99 b_k1_FDX_s10 - 1.01 s_k1_FDX_s10  = 0
 w_k1_FDX_s11_return_constraint: 0.848663 w_k1_FDX_s1 - w_k1_FDX_s11
                                 + 0.99 b_k1_FDX_s11 - 1.01 s_k1_FDX_s11  = 0
 w_k1_FDX_s000_return_constraint: 0.74823 w_k1_FDX_s00 - w_k1_FDX_s000
                                 + 0.99 b_k1_FDX_s000 - 1.01 s_k1_FDX_s000  = 0
 w_k1_FDX_s001_return_constraint: 1.32696 w_k1_FDX_s00 - w_k1_FDX_s001
                                 + 0.99 b_k1_FDX_s001 - 1.01 s_k1_FDX_s001  = 0
 w_k1_FDX_s002_return_constraint: 0.841257 w_k1_FDX_s00 - w_k1_FDX_s002
                                 + 0.99 b_k1_FDX_s002 - 1.01 s_k1_FDX_s002  = 0
 w_k1_FDX_s003_return_constraint: 0.998597 w_k1_FDX_s00 - w_k1_FDX_s003
                                 + 0.99 b_k1_FDX_s003 - 1.01 s_k1_FDX_s003  = 0
 w_k1_FDX_s004_return_constraint: 1.27728 w_k1_FDX_s00 - w_k1_FDX_s004
                                 + 0.99 b_k1_FDX_s004 - 1.01 s_k1_FDX_s004  = 0
 w_k1_FDX_s005_return_constraint: 1.62438 w_k1_FDX_s00 - w_k1_FDX_s005
                                 + 0.99 b_k1_FDX_s005 - 1.01 s_k1_FDX_s005  = 0
 w_k1_FDX_s006_return_constraint: 0.757732 w_k1_FDX_s00 - w_k1_FDX_s006
                                 + 0.99 b_k1_FDX_s006 - 1.01 s_k1_FDX_s006  = 0
 w_k1_FDX_s007_return_constraint: 1.1292 w_k1_FDX_s00 - w_k1_FDX_s007
                                 + 0.99 b_k1_FDX_s007 - 1.01 s_k1_FDX_s007  = 0
 w_k1_FDX_s010_return_constraint: 0.680467 w_k1_FDX_s01 - w_k1_FDX_s010
                                 + 0.99 b_k1_FDX_s010 - 1.01 s_k1_FDX_s010  = 0
 w_k1_FDX_s011_return_constraint: 0.696722 w_k1_FDX_s01 - w_k1_FDX_s011
                                 + 0.99 b_k1_FDX_s011 - 1.01 s_k1_FDX_s011  = 0
 w_k1_FDX_s012_return_constraint: 1.12103 w_k1_FDX_s01 - w_k1_FDX_s012
                                 + 0.99 b_k1_FDX_s012 - 1.01 s_k1_FDX_s012  = 0
 w_k1_FDX_s013_return_constraint: 0.977004 w_k1_FDX_s01 - w_k1_FDX_s013
                                 + 0.99 b_k1_FDX_s013 - 1.01 s_k1_FDX_s013  = 0
 w_k1_FDX_s014_return_constraint: 0.813112 w_k1_FDX_s01 - w_k1_FDX_s014
                                 + 0.99 b_k1_FDX_s014 - 1.01 s_k1_FDX_s014  = 0
 w_k1_FDX_s015_return_constraint: 1.03787 w_k1_FDX_s01 - w_k1_FDX_s015
                                 + 0.99 b_k1_FDX_s015 - 1.01 s_k1_FDX_s015  = 0
 w_k1_FDX_s016_return_constraint: 1.60424 w_k1_FDX_s01 - w_k1_FDX_s016
                                 + 0.99 b_k1_FDX_s016 - 1.01 s_k1_FDX_s016  = 0
 w_k1_FDX_s017_return_constraint: 1.41378 w_k1_FDX_s01 - w_k1_FDX_s017
                                 + 0.99 b_k1_FDX_s017 - 1.01 s_k1_FDX_s017  = 0
 w_k1_FDX_s100_return_constraint: 0.86558 w_k1_FDX_s10 - w_k1_FDX_s100
                                 + 0.99 b_k1_FDX_s100 - 1.01 s_k1_FDX_s100  = 0
 w_k1_FDX_s101_return_constraint: 0.91743 w_k1_FDX_s10 - w_k1_FDX_s101
                                 + 0.99 b_k1_FDX_s101 - 1.01 s_k1_FDX_s101  = 0
 w_k1_FDX_s102_return_constraint: 0.824263 w_k1_FDX_s10 - w_k1_FDX_s102
                                 + 0.99 b_k1_FDX_s102 - 1.01 s_k1_FDX_s102  = 0
 w_k1_FDX_s103_return_constraint: 1.1239 w_k1_FDX_s10 - w_k1_FDX_s103
                                 + 0.99 b_k1_FDX_s103 - 1.01 s_k1_FDX_s103  = 0
 w_k1_FDX_s104_return_constraint: 1.44192 w_k1_FDX_s10 - w_k1_FDX_s104
                                 + 0.99 b_k1_FDX_s104 - 1.01 s_k1_FDX_s104  = 0
 w_k1_FDX_s105_return_constraint: 0.811206 w_k1_FDX_s10 - w_k1_FDX_s105
                                 + 0.99 b_k1_FDX_s105 - 1.01 s_k1_FDX_s105  = 0
 w_k1_FDX_s106_return_constraint: 1.09436 w_k1_FDX_s10 - w_k1_FDX_s106
                                 + 0.99 b_k1_FDX_s106 - 1.01 s_k1_FDX_s106  = 0
 w_k1_FDX_s107_return_constraint: 0.722567 w_k1_FDX_s10 - w_k1_FDX_s107
                                 + 0.99 b_k1_FDX_s107 - 1.01 s_k1_FDX_s107  = 0
 w_k1_FDX_s110_return_constraint: 0.850998 w_k1_FDX_s11 - w_k1_FDX_s110
                                 + 0.99 b_k1_FDX_s110 - 1.01 s_k1_FDX_s110  = 0
 w_k1_FDX_s111_return_constraint: 0.747821 w_k1_FDX_s11 - w_k1_FDX_s111
                                 + 0.99 b_k1_FDX_s111 - 1.01 s_k1_FDX_s111  = 0
 w_k1_FDX_s112_return_constraint: 1.46567 w_k1_FDX_s11 - w_k1_FDX_s112
                                 + 0.99 b_k1_FDX_s112 - 1.01 s_k1_FDX_s112  = 0
 w_k1_FDX_s113_return_constraint: 1.18395 w_k1_FDX_s11 - w_k1_FDX_s113
                                 + 0.99 b_k1_FDX_s113 - 1.01 s_k1_FDX_s113  = 0
 w_k1_FDX_s114_return_constraint: 1.05938 w_k1_FDX_s11 - w_k1_FDX_s114
                                 + 0.99 b_k1_FDX_s114 - 1.01 s_k1_FDX_s114  = 0
 w_k1_FDX_s115_return_constraint: 0.816677 w_k1_FDX_s11 - w_k1_FDX_s115
                                 + 0.99 b_k1_FDX_s115 - 1.01 s_k1_FDX_s115  = 0
 w_k1_FDX_s116_return_constraint: 0.965575 w_k1_FDX_s11 - w_k1_FDX_s116
                                 + 0.99 b_k1_FDX_s116 - 1.01 s_k1_FDX_s116  = 0
 w_k1_FDX_s117_return_constraint: 0.970502 w_k1_FDX_s11 - w_k1_FDX_s117
                                 + 0.99 b_k1_FDX_s117 - 1.01 s_k1_FDX_s117  = 0
 w_k1_IBM_s0_return_constraint:  1.18407 w_0_IBM - w_k1_IBM_s0
                                 + 0.99 b_k1_IBM_s0 - 1.01 s_k1_IBM_s0  = 0
 w_k1_IBM_s1_return_constraint:  0.865 w_0_IBM - w_k1_IBM_s1 + 0.99 b_k1_IBM_s1
                                 - 1.01 s_k1_IBM_s1  = 0
 w_k1_IBM_s00_return_constraint: 0.824984 w_k1_IBM_s0 - w_k1_IBM_s00
                                 + 0.99 b_k1_IBM_s00 - 1.01 s_k1_IBM_s00  = 0
 w_k1_IBM_s01_return_constraint: 1.29015 w_k1_IBM_s0 - w_k1_IBM_s01
                                 + 0.99 b_k1_IBM_s01 - 1.01 s_k1_IBM_s01  = 0
 w_k1_IBM_s10_return_constraint: 1.10504 w_k1_IBM_s1 - w_k1_IBM_s10
                                 + 0.99 b_k1_IBM_s10 - 1.01 s_k1_IBM_s10  = 0
 w_k1_IBM_s11_return_constraint: 0.949043 w_k1_IBM_s1 - w_k1_IBM_s11
                                 + 0.99 b_k1_IBM_s11 - 1.01 s_k1_IBM_s11  = 0
 w_k1_IBM_s000_return_constraint: 0.948891 w_k1_IBM_s00 - w_k1_IBM_s000
                                 + 0.99 b_k1_IBM_s000 - 1.01 s_k1_IBM_s000  = 0
 w_k1_IBM_s001_return_constraint: 0.976498 w_k1_IBM_s00 - w_k1_IBM_s001
                                 + 0.99 b_k1_IBM_s001 - 1.01 s_k1_IBM_s001  = 0
 w_k1_IBM_s002_return_constraint: 1.28477 w_k1_IBM_s00 - w_k1_IBM_s002
                                 + 0.99 b_k1_IBM_s002 - 1.01 s_k1_IBM_s002  = 0
 w_k1_IBM_s003_return_constraint: 0.810383 w_k1_IBM_s00 - w_k1_IBM_s003
                                 + 0.99 b_k1_IBM_s003 - 1.01 s_k1_IBM_s003  = 0
 w_k1_IBM_s004_return_constraint: 0.776375 w_k1_IBM_s00 - w_k1_IBM_s004
                                 + 0.99 b_k1_IBM_s004 - 1.01 s_k1_IBM_s004  = 0
 w_k1_IBM_s005_return_constraint: 1.63111 w_k1_IBM_s00 - w_k1_IBM_s005
                                 + 0.99 b_k1_IBM_s005 - 1.01 s_k1_IBM_s005  = 0
 w_k1_IBM_s006_return_constraint: 0.570743 w_k1_IBM_s00 - w_k1_IBM_s006
                                 + 0.99 b_k1_IBM_s006 - 1.01 s_k1_IBM_s006  = 0
 w_k1_IBM_s007_return_constraint: 1.20596 w_k1_IBM_s00 - w_k1_IBM_s007
                                 + 0.99 b_k1_IBM_s007 - 1.01 s_k1_IBM_s007  = 0
 w_k1_IBM_s010_return_constraint: 0.914087 w_k1_IBM_s01 - w_k1_IBM_s010
                                 + 0.99 b_k1_IBM_s010 - 1.01 s_k1_IBM_s010  = 0
 w_k1_IBM_s011_return_constraint: 0.631432 w_k1_IBM_s01 - w_k1_IBM_s011
                                 + 0.99 b_k1_IBM_s011 - 1.01 s_k1_IBM_s011  = 0
 w_k1_IBM_s012_return_constraint: 1.18701 w_k1_IBM_s01 - w_k1_IBM_s012
                                 + 0.99 b_k1_IBM_s012 - 1.01 s_k1_IBM_s012  = 0
 w_k1_IBM_s013_return_constraint: 1.59692 w_k1_IBM_s01 - w_k1_IBM_s013
                                 + 0.99 b_k1_IBM_s013 - 1.01 s_k1_IBM_s013  = 0
 w_k1_IBM_s014_return_constraint: 0.987054 w_k1_IBM_s01 - w_k1_IBM_s014
                                 + 0.99 b_k1_IBM_s014 - 1.01 s_k1_IBM_s014  = 0
 w_k1_IBM_s015_return_constraint: 0.807774 w_k1_IBM_s01 - w_k1_IBM_s015
                                 + 0.99 b_k1_IBM_s015 - 1.01 s_k1_IBM_s015  = 0
 w_k1_IBM_s016_return_constraint: 1.72122 w_k1_IBM_s01 - w_k1_IBM_s016
                                 + 0.99 b_k1_IBM_s016 - 1.01 s_k1_IBM_s016  = 0
 w_k1_IBM_s017_return_constraint: 1.05978 w_k1_IBM_s01 - w_k1_IBM_s017
                                 + 0.99 b_k1_IBM_s017 - 1.01 s_k1_IBM_s017  = 0
 w_k1_IBM_s100_return_constraint: 0.96876 w_k1_IBM_s10 - w_k1_IBM_s100
                                 + 0.99 b_k1_IBM_s100 - 1.01 s_k1_IBM_s100  = 0
 w_k1_IBM_s101_return_constraint: 0.628065 w_k1_IBM_s10 - w_k1_IBM_s101
                                 + 0.99 b_k1_IBM_s101 - 1.01 s_k1_IBM_s101  = 0
 w_k1_IBM_s102_return_constraint: 0.610415 w_k1_IBM_s10 - w_k1_IBM_s102
                                 + 0.99 b_k1_IBM_s102 - 1.01 s_k1_IBM_s102  = 0
 w_k1_IBM_s103_return_constraint: 1.49365 w_k1_IBM_s10 - w_k1_IBM_s103
                                 + 0.99 b_k1_IBM_s103 - 1.01 s_k1_IBM_s103  = 0
 w_k1_IBM_s104_return_constraint: 0.882534 w_k1_IBM_s10 - w_k1_IBM_s104
                                 + 0.99 b_k1_IBM_s104 - 1.01 s_k1_IBM_s104  = 0
 w_k1_IBM_s105_return_constraint: 1.70363 w_k1_IBM_s10 - w_k1_IBM_s105
                                 + 0.99 b_k1_IBM_s105 - 1.01 s_k1_IBM_s105  = 0
 w_k1_IBM_s106_return_constraint: 0.745143 w_k1_IBM_s10 - w_k1_IBM_s106
                                 + 0.99 b_k1_IBM_s106 - 1.01 s_k1_IBM_s106  = 0
 w_k1_IBM_s107_return_constraint: 1.03775 w_k1_IBM_s10 - w_k1_IBM_s107
                                 + 0.99 b_k1_IBM_s107 - 1.01 s_k1_IBM_s107  = 0
 w_k1_IBM_s110_return_constraint: 0.78912 w_k1_IBM_s11 - w_k1_IBM_s110
                                 + 0.99 b_k1_IBM_s110 - 1.01 s_k1_IBM_s110  = 0
 w_k1_IBM_s111_return_constraint: 1.1296 w_k1_IBM_s11 - w_k1_IBM_s111
                                 + 0.99 b_k1_IBM_s111 - 1.01 s_k1_IBM_s111  = 0
 w_k1_IBM_s112_return_constraint: 1.17705 w_k1_IBM_s11 - w_k1_IBM_s112
                                 + 0.99 b_k1_IBM_s112 - 1.01 s_k1_IBM_s112  = 0
 w_k1_IBM_s113_return_constraint: 0.871435 w_k1_IBM_s11 - w_k1_IBM_s113
                                 + 0.99 b_k1_IBM_s113 - 1.01 s_k1_IBM_s113  = 0
 w_k1_IBM_s114_return_constraint: 1.5565 w_k1_IBM_s11 - w_k1_IBM_s114
                                 + 0.99 b_k1_IBM_s114 - 1.01 s_k1_IBM_s114  = 0
 w_k1_IBM_s115_return_constraint: 0.60809 w_k1_IBM_s11 - w_k1_IBM_s115
                                 + 0.99 b_k1_IBM_s115 - 1.01 s_k1_IBM_s115  = 0
 w_k1_IBM_s116_return_constraint: 0.526911 w_k1_IBM_s11 - w_k1_IBM_s116
                                 + 0.99 b_k1_IBM_s116 - 1.01 s_k1_IBM_s116  = 0
 w_k1_IBM_s117_return_constraint: 0.967648 w_k1_IBM_s11 - w_k1_IBM_s117
                                 + 0.99 b_k1_IBM_s117 - 1.01 s_k1_IBM_s117  = 0
 w_k1_INTC_s0_return_constraint: 1.09448 w_0_INTC - w_k1_INTC_s0
                                 + 0.99 b_k1_INTC_s0 - 1.01 s_k1_INTC_s0  = 0
 w_k1_INTC_s1_return_constraint: 0.936275 w_0_INTC - w_k1_INTC_s1
                                 + 0.99 b_k1_INTC_s1 - 1.01 s_k1_INTC_s1  = 0
 w_k1_INTC_s00_return_constraint: 0.935831 w_k1_INTC_s0 - w_k1_INTC_s00
                                 + 0.99 b_k1_INTC_s00 - 1.01 s_k1_INTC_s00  = 0
 w_k1_INTC_s01_return_constraint: 0.957618 w_k1_INTC_s0 - w_k1_INTC_s01
                                 + 0.99 b_k1_INTC_s01 - 1.01 s_k1_INTC_s01  = 0
 w_k1_INTC_s10_return_constraint: 1.20326 w_k1_INTC_s1 - w_k1_INTC_s10
                                 + 0.99 b_k1_INTC_s10 - 1.01 s_k1_INTC_s10  = 0
 w_k1_INTC_s11_return_constraint: 0.882826 w_k1_INTC_s1 - w_k1_INTC_s11
                                 + 0.99 b_k1_INTC_s11 - 1.01 s_k1_INTC_s11  = 0
 w_k1_INTC_s000_return_constraint: 0.781154 w_k1_INTC_s00 - w_k1_INTC_s000
                                 + 0.99 b_k1_INTC_s000 - 1.01 s_k1_INTC_s000
                                  = 0
 w_k1_INTC_s001_return_constraint: 1.27619 w_k1_INTC_s00 - w_k1_INTC_s001
                                 + 0.99 b_k1_INTC_s001 - 1.01 s_k1_INTC_s001
                                  = 0
 w_k1_INTC_s002_return_constraint: 0.973732 w_k1_INTC_s00 - w_k1_INTC_s002
                                 + 0.99 b_k1_INTC_s002 - 1.01 s_k1_INTC_s002
                                  = 0
 w_k1_INTC_s003_return_constraint: 0.955331 w_k1_INTC_s00 - w_k1_INTC_s003
                                 + 0.99 b_k1_INTC_s003 - 1.01 s_k1_INTC_s003
                                  = 0
 w_k1_INTC_s004_return_constraint: 1.12058 w_k1_INTC_s00 - w_k1_INTC_s004
                                 + 0.99 b_k1_INTC_s004 - 1.01 s_k1_INTC_s004
                                  = 0
 w_k1_INTC_s005_return_constraint: 1.48103 w_k1_INTC_s00 - w_k1_INTC_s005
                                 + 0.99 b_k1_INTC_s005 - 1.01 s_k1_INTC_s005
                                  = 0
 w_k1_INTC_s006_return_constraint: 0.881621 w_k1_INTC_s00 - w_k1_INTC_s006
                                 + 0.99 b_k1_INTC_s006 - 1.01 s_k1_INTC_s006
                                  = 0
 w_k1_INTC_s007_return_constraint: 1.18477 w_k1_INTC_s00 - w_k1_INTC_s007
                                 + 0.99 b_k1_INTC_s007 - 1.01 s_k1_INTC_s007
                                  = 0
 w_k1_INTC_s010_return_constraint: 0.876411 w_k1_INTC_s01 - w_k1_INTC_s010
                                 + 0.99 b_k1_INTC_s010 - 1.01 s_k1_INTC_s010
                                  = 0
 w_k1_INTC_s011_return_constraint: 0.854295 w_k1_INTC_s01 - w_k1_INTC_s011
                                 + 0.99 b_k1_INTC_s011 - 1.01 s_k1_INTC_s011
                                  = 0
 w_k1_INTC_s012_return_constraint: 1.30702 w_k1_INTC_s01 - w_k1_INTC_s012
                                 + 0.99 b_k1_INTC_s012 - 1.01 s_k1_INTC_s012
                                  = 0
 w_k1_INTC_s013_return_constraint: 1.1352 w_k1_INTC_s01 - w_k1_INTC_s013
                                 + 0.99 b_k1_INTC_s013 - 1.01 s_k1_INTC_s013
                                  = 0
 w_k1_INTC_s014_return_constraint: 0.929048 w_k1_INTC_s01 - w_k1_INTC_s014
                                 + 0.99 b_k1_INTC_s014 - 1.01 s_k1_INTC_s014
                                  = 0
 w_k1_INTC_s015_return_constraint: 0.928084 w_k1_INTC_s01 - w_k1_INTC_s015
                                 + 0.99 b_k1_INTC_s015 - 1.01 s_k1_INTC_s015
                                  = 0
 w_k1_INTC_s016_return_constraint: 1.54369 w_k1_INTC_s01 - w_k1_INTC_s016
                                 + 0.99 b_k1_INTC_s016 - 1.01 s_k1_INTC_s016
                                  = 0
 w_k1_INTC_s017_return_constraint: 1.404 w_k1_INTC_s01 - w_k1_INTC_s017
                                 + 0.99 b_k1_INTC_s017 - 1.01 s_k1_INTC_s017
                                  = 0
 w_k1_INTC_s100_return_constraint: 0.862442 w_k1_INTC_s10 - w_k1_INTC_s100
                                 + 0.99 b_k1_INTC_s100 - 1.01 s_k1_INTC_s100
                                  = 0
 w_k1_INTC_s101_return_constraint: 0.782856 w_k1_INTC_s10 - w_k1_INTC_s101
                                 + 0.99 b_k1_INTC_s101 - 1.01 s_k1_INTC_s101
                                  = 0
 w_k1_INTC_s102_return_constraint: 0.767007 w_k1_INTC_s10 - w_k1_INTC_s102
                                 + 0.99 b_k1_INTC_s102 - 1.01 s_k1_INTC_s102
                                  = 0
 w_k1_INTC_s103_return_constraint: 0.984878 w_k1_INTC_s10 - w_k1_INTC_s103
                                 + 0.99 b_k1_INTC_s103 - 1.01 s_k1_INTC_s103
                                  = 0
 w_k1_INTC_s104_return_constraint: 1.14937 w_k1_INTC_s10 - w_k1_INTC_s104
                                 + 0.99 b_k1_INTC_s104 - 1.01 s_k1_INTC_s104
                                  = 0
 w_k1_INTC_s105_return_constraint: 0.865968 w_k1_INTC_s10 - w_k1_INTC_s105
                                 + 0.99 b_k1_INTC_s105 - 1.01 s_k1_INTC_s105
                                  = 0
 w_k1_INTC_s106_return_constraint: 0.890147 w_k1_INTC_s10 - w_k1_INTC_s106
                                 + 0.99 b_k1_INTC_s106 - 1.01 s_k1_INTC_s106
                                  = 0
 w_k1_INTC_s107_return_constraint: 0.834597 w_k1_INTC_s10 - w_k1_INTC_s107
                                 + 0.99 b_k1_INTC_s107 - 1.01 s_k1_INTC_s107
                                  = 0
 w_k1_INTC_s110_return_constraint: 0.835714 w_k1_INTC_s11 - w_k1_INTC_s110
                                 + 0.99 b_k1_INTC_s110 - 1.01 s_k1_INTC_s110
                                  = 0
 w_k1_INTC_s111_return_constraint: 0.879686 w_k1_INTC_s11 - w_k1_INTC_s111
                                 + 0.99 b_k1_INTC_s111 - 1.01 s_k1_INTC_s111
                                  = 0
 w_k1_INTC_s112_return_constraint: 1.12161 w_k1_INTC_s11 - w_k1_INTC_s112
                                 + 0.99 b_k1_INTC_s112 - 1.01 s_k1_INTC_s112
                                  = 0
 w_k1_INTC_s113_return_constraint: 1.03147 w_k1_INTC_s11 - w_k1_INTC_s113
                                 + 0.99 b_k1_INTC_s113 - 1.01 s_k1_INTC_s113
                                  = 0
 w_k1_INTC_s114_return_constraint: 0.995882 w_k1_INTC_s11 - w_k1_INTC_s114
                                 + 0.99 b_k1_INTC_s114 - 1.01 s_k1_INTC_s114
                                  = 0
 w_k1_INTC_s115_return_constraint: 0.854941 w_k1_INTC_s11 - w_k1_INTC_s115
                                 + 0.99 b_k1_INTC_s115 - 1.01 s_k1_INTC_s115
                                  = 0
 w_k1_INTC_s116_return_constraint: 0.854649 w_k1_INTC_s11 - w_k1_INTC_s116
                                 + 0.99 b_k1_INTC_s116 - 1.01 s_k1_INTC_s116
                                  = 0
 w_k1_INTC_s117_return_constraint: 0.92927 w_k1_INTC_s11 - w_k1_INTC_s117
                                 + 0.99 b_k1_INTC_s117 - 1.01 s_k1_INTC_s117
                                  = 0
 w_k1_KO_s0_return_constraint:   1.14885 w_0_KO - w_k1_KO_s0 + 0.99 b_k1_KO_s0
                                 - 1.01 s_k1_KO_s0  = 0
 w_k1_KO_s1_return_constraint:   0.858881 w_0_KO - w_k1_KO_s1 + 0.99 b_k1_KO_s1
                                 - 1.01 s_k1_KO_s1  = 0
 w_k1_KO_s00_return_constraint:  0.969943 w_k1_KO_s0 - w_k1_KO_s00
                                 + 0.99 b_k1_KO_s00 - 1.01 s_k1_KO_s00  = 0
 w_k1_KO_s01_return_constraint:  1.20278 w_k1_KO_s0 - w_k1_KO_s01
                                 + 0.99 b_k1_KO_s01 - 1.01 s_k1_KO_s01  = 0
 w_k1_KO_s10_return_constraint:  1.17591 w_k1_KO_s1 - w_k1_KO_s10
                                 + 0.99 b_k1_KO_s10 - 1.01 s_k1_KO_s10  = 0
 w_k1_KO_s11_return_constraint:  0.878424 w_k1_KO_s1 - w_k1_KO_s11
                                 + 0.99 b_k1_KO_s11 - 1.01 s_k1_KO_s11  = 0
 w_k1_KO_s000_return_constraint: 0.835894 w_k1_KO_s00 - w_k1_KO_s000
                                 + 0.99 b_k1_KO_s000 - 1.01 s_k1_KO_s000  = 0
 w_k1_KO_s001_return_constraint: 1.0015 w_k1_KO_s00 - w_k1_KO_s001
                                 + 0.99 b_k1_KO_s001 - 1.01 s_k1_KO_s001  = 0
 w_k1_KO_s002_return_constraint: 1.04981 w_k1_KO_s00 - w_k1_KO_s002
                                 + 0.99 b_k1_KO_s002 - 1.01 s_k1_KO_s002  = 0
 w_k1_KO_s003_return_constraint: 0.800131 w_k1_KO_s00 - w_k1_KO_s003
                                 + 0.99 b_k1_KO_s003 - 1.01 s_k1_KO_s003  = 0
 w_k1_KO_s004_return_constraint: 1.07139 w_k1_KO_s00 - w_k1_KO_s004
                                 + 0.99 b_k1_KO_s004 - 1.01 s_k1_KO_s004  = 0
 w_k1_KO_s005_return_constraint: 1.67107 w_k1_KO_s00 - w_k1_KO_s005
                                 + 0.99 b_k1_KO_s005 - 1.01 s_k1_KO_s005  = 0
 w_k1_KO_s006_return_constraint: 0.665007 w_k1_KO_s00 - w_k1_KO_s006
                                 + 0.99 b_k1_KO_s006 - 1.01 s_k1_KO_s006  = 0
 w_k1_KO_s007_return_constraint: 1.16629 w_k1_KO_s00 - w_k1_KO_s007
                                 + 0.99 b_k1_KO_s007 - 1.01 s_k1_KO_s007  = 0
 w_k1_KO_s010_return_constraint: 0.756041 w_k1_KO_s01 - w_k1_KO_s010
                                 + 0.99 b_k1_KO_s010 - 1.01 s_k1_KO_s010  = 0
 w_k1_KO_s011_return_constraint: 0.575564 w_k1_KO_s01 - w_k1_KO_s011
                                 + 0.99 b_k1_KO_s011 - 1.01 s_k1_KO_s011  = 0
 w_k1_KO_s012_return_constraint: 1.12444 w_k1_KO_s01 - w_k1_KO_s012
                                 + 0.99 b_k1_KO_s012 - 1.01 s_k1_KO_s012  = 0
 w_k1_KO_s013_return_constraint: 1.18289 w_k1_KO_s01 - w_k1_KO_s013
                                 + 0.99 b_k1_KO_s013 - 1.01 s_k1_KO_s013  = 0
 w_k1_KO_s014_return_constraint: 0.797902 w_k1_KO_s01 - w_k1_KO_s014
                                 + 0.99 b_k1_KO_s014 - 1.01 s_k1_KO_s014  = 0
 w_k1_KO_s015_return_constraint: 0.827197 w_k1_KO_s01 - w_k1_KO_s015
                                 + 0.99 b_k1_KO_s015 - 1.01 s_k1_KO_s015  = 0
 w_k1_KO_s016_return_constraint: 1.59837 w_k1_KO_s01 - w_k1_KO_s016
                                 + 0.99 b_k1_KO_s016 - 1.01 s_k1_KO_s016  = 0
 w_k1_KO_s017_return_constraint: 1.22922 w_k1_KO_s01 - w_k1_KO_s017
                                 + 0.99 b_k1_KO_s017 - 1.01 s_k1_KO_s017  = 0
 w_k1_KO_s100_return_constraint: 0.932588 w_k1_KO_s10 - w_k1_KO_s100
                                 + 0.99 b_k1_KO_s100 - 1.01 s_k1_KO_s100  = 0
 w_k1_KO_s101_return_constraint: 0.682599 w_k1_KO_s10 - w_k1_KO_s101
                                 + 0.99 b_k1_KO_s101 - 1.01 s_k1_KO_s101  = 0
 w_k1_KO_s102_return_constraint: 0.755971 w_k1_KO_s10 - w_k1_KO_s102
                                 + 0.99 b_k1_KO_s102 - 1.01 s_k1_KO_s102  = 0
 w_k1_KO_s103_return_constraint: 1.26397 w_k1_KO_s10 - w_k1_KO_s103
                                 + 0.99 b_k1_KO_s103 - 1.01 s_k1_KO_s103  = 0
 w_k1_KO_s104_return_constraint: 1.04079 w_k1_KO_s10 - w_k1_KO_s104
                                 + 0.99 b_k1_KO_s104 - 1.01 s_k1_KO_s104  = 0
 w_k1_KO_s105_return_constraint: 1.08556 w_k1_KO_s10 - w_k1_KO_s105
                                 + 0.99 b_k1_KO_s105 - 1.01 s_k1_KO_s105  = 0
 w_k1_KO_s106_return_constraint: 0.888612 w_k1_KO_s10 - w_k1_KO_s106
                                 + 0.99 b_k1_KO_s106 - 1.01 s_k1_KO_s106  = 0
 w_k1_KO_s107_return_constraint: 0.829689 w_k1_KO_s10 - w_k1_KO_s107
                                 + 0.99 b_k1_KO_s107 - 1.01 s_k1_KO_s107  = 0
 w_k1_KO_s110_return_constraint: 0.905858 w_k1_KO_s11 - w_k1_KO_s110
                                 + 0.99 b_k1_KO_s110 - 1.01 s_k1_KO_s110  = 0
 w_k1_KO_s111_return_constraint: 0.925609 w_k1_KO_s11 - w_k1_KO_s111
                                 + 0.99 b_k1_KO_s111 - 1.01 s_k1_KO_s111  = 0
 w_k1_KO_s112_return_constraint: 1.22155 w_k1_KO_s11 - w_k1_KO_s112
                                 + 0.99 b_k1_KO_s112 - 1.01 s_k1_KO_s112  = 0
 w_k1_KO_s113_return_constraint: 1.0111 w_k1_KO_s11 - w_k1_KO_s113
                                 + 0.99 b_k1_KO_s113 - 1.01 s_k1_KO_s113  = 0
 w_k1_KO_s114_return_constraint: 1.23054 w_k1_KO_s11 - w_k1_KO_s114
                                 + 0.99 b_k1_KO_s114 - 1.01 s_k1_KO_s114  = 0
 w_k1_KO_s115_return_constraint: 0.65278 w_k1_KO_s11 - w_k1_KO_s115
                                 + 0.99 b_k1_KO_s115 - 1.01 s_k1_KO_s115  = 0
 w_k1_KO_s116_return_constraint: 0.732642 w_k1_KO_s11 - w_k1_KO_s116
                                 + 0.99 b_k1_KO_s116 - 1.01 s_k1_KO_s116  = 0
 w_k1_KO_s117_return_constraint: 0.914785 w_k1_KO_s11 - w_k1_KO_s117
                                 + 0.99 b_k1_KO_s117 - 1.01 s_k1_KO_s117  = 0
 w_k1_MAT_s0_return_constraint:  1.24451 w_0_MAT - w_k1_MAT_s0
                                 + 0.99 b_k1_MAT_s0 - 1.01 s_k1_MAT_s0  = 0
 w_k1_MAT_s1_return_constraint:  0.843306 w_0_MAT - w_k1_MAT_s1
                                 + 0.99 b_k1_MAT_s1 - 1.01 s_k1_MAT_s1  = 0
 w_k1_MAT_s00_return_constraint: 0.767175 w_k1_MAT_s0 - w_k1_MAT_s00
                                 + 0.99 b_k1_MAT_s00 - 1.01 s_k1_MAT_s00  = 0
 w_k1_MAT_s01_return_constraint: 1.21861 w_k1_MAT_s0 - w_k1_MAT_s01
                                 + 0.99 b_k1_MAT_s01 - 1.01 s_k1_MAT_s01  = 0
 w_k1_MAT_s10_return_constraint: 0.979716 w_k1_MAT_s1 - w_k1_MAT_s10
                                 + 0.99 b_k1_MAT_s10 - 1.01 s_k1_MAT_s10  = 0
 w_k1_MAT_s11_return_constraint: 0.957185 w_k1_MAT_s1 - w_k1_MAT_s11
                                 + 0.99 b_k1_MAT_s11 - 1.01 s_k1_MAT_s11  = 0
 w_k1_MAT_s000_return_constraint: 1.0949 w_k1_MAT_s00 - w_k1_MAT_s000
                                 + 0.99 b_k1_MAT_s000 - 1.01 s_k1_MAT_s000  = 0
 w_k1_MAT_s001_return_constraint: 1.05089 w_k1_MAT_s00 - w_k1_MAT_s001
                                 + 0.99 b_k1_MAT_s001 - 1.01 s_k1_MAT_s001  = 0
 w_k1_MAT_s002_return_constraint: 1.45528 w_k1_MAT_s00 - w_k1_MAT_s002
                                 + 0.99 b_k1_MAT_s002 - 1.01 s_k1_MAT_s002  = 0
 w_k1_MAT_s003_return_constraint: 1.15504 w_k1_MAT_s00 - w_k1_MAT_s003
                                 + 0.99 b_k1_MAT_s003 - 1.01 s_k1_MAT_s003  = 0
 w_k1_MAT_s004_return_constraint: 1.12168 w_k1_MAT_s00 - w_k1_MAT_s004
                                 + 0.99 b_k1_MAT_s004 - 1.01 s_k1_MAT_s004  = 0
 w_k1_MAT_s005_return_constraint: 2.20918 w_k1_MAT_s00 - w_k1_MAT_s005
                                 + 0.99 b_k1_MAT_s005 - 1.01 s_k1_MAT_s005  = 0
 w_k1_MAT_s006_return_constraint: 0.666849 w_k1_MAT_s00 - w_k1_MAT_s006
                                 + 0.99 b_k1_MAT_s006 - 1.01 s_k1_MAT_s006  = 0
 w_k1_MAT_s007_return_constraint: 1.5855 w_k1_MAT_s00 - w_k1_MAT_s007
                                 + 0.99 b_k1_MAT_s007 - 1.01 s_k1_MAT_s007  = 0
 w_k1_MAT_s010_return_constraint: 0.776317 w_k1_MAT_s01 - w_k1_MAT_s010
                                 + 0.99 b_k1_MAT_s010 - 1.01 s_k1_MAT_s010  = 0
 w_k1_MAT_s011_return_constraint: 0.633033 w_k1_MAT_s01 - w_k1_MAT_s011
                                 + 0.99 b_k1_MAT_s011 - 1.01 s_k1_MAT_s011  = 0
 w_k1_MAT_s012_return_constraint: 1.23872 w_k1_MAT_s01 - w_k1_MAT_s012
                                 + 0.99 b_k1_MAT_s012 - 1.01 s_k1_MAT_s012  = 0
 w_k1_MAT_s013_return_constraint: 1.62994 w_k1_MAT_s01 - w_k1_MAT_s013
                                 + 0.99 b_k1_MAT_s013 - 1.01 s_k1_MAT_s013  = 0
 w_k1_MAT_s014_return_constraint: 1.01153 w_k1_MAT_s01 - w_k1_MAT_s014
                                 + 0.99 b_k1_MAT_s014 - 1.01 s_k1_MAT_s014  = 0
 w_k1_MAT_s015_return_constraint: 0.852638 w_k1_MAT_s01 - w_k1_MAT_s015
                                 + 0.99 b_k1_MAT_s015 - 1.01 s_k1_MAT_s015  = 0
 w_k1_MAT_s016_return_constraint: 2.21617 w_k1_MAT_s01 - w_k1_MAT_s016
                                 + 0.99 b_k1_MAT_s016 - 1.01 s_k1_MAT_s016  = 0
 w_k1_MAT_s017_return_constraint: 1.22345 w_k1_MAT_s01 - w_k1_MAT_s017
                                 + 0.99 b_k1_MAT_s017 - 1.01 s_k1_MAT_s017  = 0
 w_k1_MAT_s100_return_constraint: 1.07939 w_k1_MAT_s10 - w_k1_MAT_s100
                                 + 0.99 b_k1_MAT_s100 - 1.01 s_k1_MAT_s100  = 0
 w_k1_MAT_s101_return_constraint: 0.896896 w_k1_MAT_s10 - w_k1_MAT_s101
                                 + 0.99 b_k1_MAT_s101 - 1.01 s_k1_MAT_s101  = 0
 w_k1_MAT_s102_return_constraint: 0.604951 w_k1_MAT_s10 - w_k1_MAT_s102
                                 + 0.99 b_k1_MAT_s102 - 1.01 s_k1_MAT_s102  = 0
 w_k1_MAT_s103_return_constraint: 1.84324 w_k1_MAT_s10 - w_k1_MAT_s103
                                 + 0.99 b_k1_MAT_s103 - 1.01 s_k1_MAT_s103  = 0
 w_k1_MAT_s104_return_constraint: 1.32365 w_k1_MAT_s10 - w_k1_MAT_s104
                                 + 0.99 b_k1_MAT_s104 - 1.01 s_k1_MAT_s104  = 0
 w_k1_MAT_s105_return_constraint: 1.56657 w_k1_MAT_s10 - w_k1_MAT_s105
                                 + 0.99 b_k1_MAT_s105 - 1.01 s_k1_MAT_s105  = 0
 w_k1_MAT_s106_return_constraint: 0.753132 w_k1_MAT_s10 - w_k1_MAT_s106
                                 + 0.99 b_k1_MAT_s106 - 1.01 s_k1_MAT_s106  = 0
 w_k1_MAT_s107_return_constraint: 1.24928 w_k1_MAT_s10 - w_k1_MAT_s107
                                 + 0.99 b_k1_MAT_s107 - 1.01 s_k1_MAT_s107  = 0
 w_k1_MAT_s110_return_constraint: 0.58117 w_k1_MAT_s11 - w_k1_MAT_s110
                                 + 0.99 b_k1_MAT_s110 - 1.01 s_k1_MAT_s110  = 0
 w_k1_MAT_s111_return_constraint: 0.959495 w_k1_MAT_s11 - w_k1_MAT_s111
                                 + 0.99 b_k1_MAT_s111 - 1.01 s_k1_MAT_s111  = 0
 w_k1_MAT_s112_return_constraint: 1.03325 w_k1_MAT_s11 - w_k1_MAT_s112
                                 + 0.99 b_k1_MAT_s112 - 1.01 s_k1_MAT_s112  = 0
 w_k1_MAT_s113_return_constraint: 0.86075 w_k1_MAT_s11 - w_k1_MAT_s113
                                 + 0.99 b_k1_MAT_s113 - 1.01 s_k1_MAT_s113  = 0
 w_k1_MAT_s114_return_constraint: 1.71123 w_k1_MAT_s11 - w_k1_MAT_s114
                                 + 0.99 b_k1_MAT_s114 - 1.01 s_k1_MAT_s114  = 0
 w_k1_MAT_s115_return_constraint: 0.564148 w_k1_MAT_s11 - w_k1_MAT_s115
                                 + 0.99 b_k1_MAT_s115 - 1.01 s_k1_MAT_s115  = 0
 w_k1_MAT_s116_return_constraint: 0.580303 w_k1_MAT_s11 - w_k1_MAT_s116
                                 + 0.99 b_k1_MAT_s116 - 1.01 s_k1_MAT_s116  = 0
 w_k1_MAT_s117_return_constraint: 1.12023 w_k1_MAT_s11 - w_k1_MAT_s117
                                 + 0.99 b_k1_MAT_s117 - 1.01 s_k1_MAT_s117  = 0
 w_k1_MSFT_s0_return_constraint: 1.11403 w_0_MSFT - w_k1_MSFT_s0
                                 + 0.99 b_k1_MSFT_s0 - 1.01 s_k1_MSFT_s0  = 0
 w_k1_MSFT_s1_return_constraint: 0.890161 w_0_MSFT - w_k1_MSFT_s1
                                 + 0.99 b_k1_MSFT_s1 - 1.01 s_k1_MSFT_s1  = 0
 w_k1_MSFT_s00_return_constraint: 1.01309 w_k1_MSFT_s0 - w_k1_MSFT_s00
                                 + 0.99 b_k1_MSFT_s00 - 1.01 s_k1_MSFT_s00  = 0
 w_k1_MSFT_s01_return_constraint: 0.970658 w_k1_MSFT_s0 - w_k1_MSFT_s01
                                 + 0.99 b_k1_MSFT_s01 - 1.01 s_k1_MSFT_s01  = 0
 w_k1_MSFT_s10_return_constraint: 1.18346 w_k1_MSFT_s1 - w_k1_MSFT_s10
                                 + 0.99 b_k1_MSFT_s10 - 1.01 s_k1_MSFT_s10  = 0
 w_k1_MSFT_s11_return_constraint: 0.924472 w_k1_MSFT_s1 - w_k1_MSFT_s11
                                 + 0.99 b_k1_MSFT_s11 - 1.01 s_k1_MSFT_s11  = 0
 w_k1_MSFT_s000_return_constraint: 0.766245 w_k1_MSFT_s00 - w_k1_MSFT_s000
                                 + 0.99 b_k1_MSFT_s000 - 1.01 s_k1_MSFT_s000
                                  = 0
 w_k1_MSFT_s001_return_constraint: 1.08716 w_k1_MSFT_s00 - w_k1_MSFT_s001
                                 + 0.99 b_k1_MSFT_s001 - 1.01 s_k1_MSFT_s001
                                  = 0
 w_k1_MSFT_s002_return_constraint: 0.92445 w_k1_MSFT_s00 - w_k1_MSFT_s002
                                 + 0.99 b_k1_MSFT_s002 - 1.01 s_k1_MSFT_s002
                                  = 0
 w_k1_MSFT_s003_return_constraint: 0.904512 w_k1_MSFT_s00 - w_k1_MSFT_s003
                                 + 0.99 b_k1_MSFT_s003 - 1.01 s_k1_MSFT_s003
                                  = 0
 w_k1_MSFT_s004_return_constraint: 1.28047 w_k1_MSFT_s00 - w_k1_MSFT_s004
                                 + 0.99 b_k1_MSFT_s004 - 1.01 s_k1_MSFT_s004
                                  = 0
 w_k1_MSFT_s005_return_constraint: 1.45993 w_k1_MSFT_s00 - w_k1_MSFT_s005
                                 + 0.99 b_k1_MSFT_s005 - 1.01 s_k1_MSFT_s005
                                  = 0
 w_k1_MSFT_s006_return_constraint: 0.800729 w_k1_MSFT_s00 - w_k1_MSFT_s006
                                 + 0.99 b_k1_MSFT_s006 - 1.01 s_k1_MSFT_s006
                                  = 0
 w_k1_MSFT_s007_return_constraint: 1.18302 w_k1_MSFT_s00 - w_k1_MSFT_s007
                                 + 0.99 b_k1_MSFT_s007 - 1.01 s_k1_MSFT_s007
                                  = 0
 w_k1_MSFT_s010_return_constraint: 0.854734 w_k1_MSFT_s01 - w_k1_MSFT_s010
                                 + 0.99 b_k1_MSFT_s010 - 1.01 s_k1_MSFT_s010
                                  = 0
 w_k1_MSFT_s011_return_constraint: 0.717756 w_k1_MSFT_s01 - w_k1_MSFT_s011
                                 + 0.99 b_k1_MSFT_s011 - 1.01 s_k1_MSFT_s011
                                  = 0
 w_k1_MSFT_s012_return_constraint: 1.18877 w_k1_MSFT_s01 - w_k1_MSFT_s012
                                 + 0.99 b_k1_MSFT_s012 - 1.01 s_k1_MSFT_s012
                                  = 0
 w_k1_MSFT_s013_return_constraint: 1.00371 w_k1_MSFT_s01 - w_k1_MSFT_s013
                                 + 0.99 b_k1_MSFT_s013 - 1.01 s_k1_MSFT_s013
                                  = 0
 w_k1_MSFT_s014_return_constraint: 0.874402 w_k1_MSFT_s01 - w_k1_MSFT_s014
                                 + 0.99 b_k1_MSFT_s014 - 1.01 s_k1_MSFT_s014
                                  = 0
 w_k1_MSFT_s015_return_constraint: 1.04974 w_k1_MSFT_s01 - w_k1_MSFT_s015
                                 + 0.99 b_k1_MSFT_s015 - 1.01 s_k1_MSFT_s015
                                  = 0
 w_k1_MSFT_s016_return_constraint: 1.60144 w_k1_MSFT_s01 - w_k1_MSFT_s016
                                 + 0.99 b_k1_MSFT_s016 - 1.01 s_k1_MSFT_s016
                                  = 0
 w_k1_MSFT_s017_return_constraint: 1.26329 w_k1_MSFT_s01 - w_k1_MSFT_s017
                                 + 0.99 b_k1_MSFT_s017 - 1.01 s_k1_MSFT_s017
                                  = 0
 w_k1_MSFT_s100_return_constraint: 0.887586 w_k1_MSFT_s10 - w_k1_MSFT_s100
                                 + 0.99 b_k1_MSFT_s100 - 1.01 s_k1_MSFT_s100
                                  = 0
 w_k1_MSFT_s101_return_constraint: 0.795436 w_k1_MSFT_s10 - w_k1_MSFT_s101
                                 + 0.99 b_k1_MSFT_s101 - 1.01 s_k1_MSFT_s101
                                  = 0
 w_k1_MSFT_s102_return_constraint: 0.91224 w_k1_MSFT_s10 - w_k1_MSFT_s102
                                 + 0.99 b_k1_MSFT_s102 - 1.01 s_k1_MSFT_s102
                                  = 0
 w_k1_MSFT_s103_return_constraint: 1.03922 w_k1_MSFT_s10 - w_k1_MSFT_s103
                                 + 0.99 b_k1_MSFT_s103 - 1.01 s_k1_MSFT_s103
                                  = 0
 w_k1_MSFT_s104_return_constraint: 1.22603 w_k1_MSFT_s10 - w_k1_MSFT_s104
                                 + 0.99 b_k1_MSFT_s104 - 1.01 s_k1_MSFT_s104
                                  = 0
 w_k1_MSFT_s105_return_constraint: 0.90689 w_k1_MSFT_s10 - w_k1_MSFT_s105
                                 + 0.99 b_k1_MSFT_s105 - 1.01 s_k1_MSFT_s105
                                  = 0
 w_k1_MSFT_s106_return_constraint: 1.05589 w_k1_MSFT_s10 - w_k1_MSFT_s106
                                 + 0.99 b_k1_MSFT_s106 - 1.01 s_k1_MSFT_s106
                                  = 0
 w_k1_MSFT_s107_return_constraint: 0.77018 w_k1_MSFT_s10 - w_k1_MSFT_s107
                                 + 0.99 b_k1_MSFT_s107 - 1.01 s_k1_MSFT_s107
                                  = 0
 w_k1_MSFT_s110_return_constraint: 0.800182 w_k1_MSFT_s11 - w_k1_MSFT_s110
                                 + 0.99 b_k1_MSFT_s110 - 1.01 s_k1_MSFT_s110
                                  = 0
 w_k1_MSFT_s111_return_constraint: 0.838075 w_k1_MSFT_s11 - w_k1_MSFT_s111
                                 + 0.99 b_k1_MSFT_s111 - 1.01 s_k1_MSFT_s111
                                  = 0
 w_k1_MSFT_s112_return_constraint: 1.21567 w_k1_MSFT_s11 - w_k1_MSFT_s112
                                 + 0.99 b_k1_MSFT_s112 - 1.01 s_k1_MSFT_s112
                                  = 0
 w_k1_MSFT_s113_return_constraint: 1.103 w_k1_MSFT_s11 - w_k1_MSFT_s113
                                 + 0.99 b_k1_MSFT_s113 - 1.01 s_k1_MSFT_s113
                                  = 0
 w_k1_MSFT_s114_return_constraint: 0.974843 w_k1_MSFT_s11 - w_k1_MSFT_s114
                                 + 0.99 b_k1_MSFT_s114 - 1.01 s_k1_MSFT_s114
                                  = 0
 w_k1_MSFT_s115_return_constraint: 0.789961 w_k1_MSFT_s11 - w_k1_MSFT_s115
                                 + 0.99 b_k1_MSFT_s115 - 1.01 s_k1_MSFT_s115
                                  = 0
 w_k1_MSFT_s116_return_constraint: 0.929747 w_k1_MSFT_s11 - w_k1_MSFT_s116
                                 + 0.99 b_k1_MSFT_s116 - 1.01 s_k1_MSFT_s116
                                  = 0
 w_k1_MSFT_s117_return_constraint: 0.908675 w_k1_MSFT_s11 - w_k1_MSFT_s117
                                 + 0.99 b_k1_MSFT_s117 - 1.01 s_k1_MSFT_s117
                                  = 0
 w_k1_PG_s0_return_constraint:   1.0775 w_0_PG - w_k1_PG_s0 + 0.99 b_k1_PG_s0
                                 - 1.01 s_k1_PG_s0  = 0
 w_k1_PG_s1_return_constraint:   0.891025 w_0_PG - w_k1_PG_s1 + 0.99 b_k1_PG_s1
                                 - 1.01 s_k1_PG_s1  = 0
 w_k1_PG_s00_return_constraint:  1.00922 w_k1_PG_s0 - w_k1_PG_s00
                                 + 0.99 b_k1_PG_s00 - 1.01 s_k1_PG_s00  = 0
 w_k1_PG_s01_return_constraint:  1.07568 w_k1_PG_s0 - w_k1_PG_s01
                                 + 0.99 b_k1_PG_s01 - 1.01 s_k1_PG_s01  = 0
 w_k1_PG_s10_return_constraint:  1.13781 w_k1_PG_s1 - w_k1_PG_s10
                                 + 0.99 b_k1_PG_s10 - 1.01 s_k1_PG_s10  = 0
 w_k1_PG_s11_return_constraint:  0.903501 w_k1_PG_s1 - w_k1_PG_s11
                                 + 0.99 b_k1_PG_s11 - 1.01 s_k1_PG_s11  = 0
 w_k1_PG_s000_return_constraint: 0.825734 w_k1_PG_s00 - w_k1_PG_s000
                                 + 0.99 b_k1_PG_s000 - 1.01 s_k1_PG_s000  = 0
 w_k1_PG_s001_return_constraint: 1.03732 w_k1_PG_s00 - w_k1_PG_s001
                                 + 0.99 b_k1_PG_s001 - 1.01 s_k1_PG_s001  = 0
 w_k1_PG_s002_return_constraint: 0.979168 w_k1_PG_s00 - w_k1_PG_s002
                                 + 0.99 b_k1_PG_s002 - 1.01 s_k1_PG_s002  = 0
 w_k1_PG_s003_return_constraint: 0.993715 w_k1_PG_s00 - w_k1_PG_s003
                                 + 0.99 b_k1_PG_s003 - 1.01 s_k1_PG_s003  = 0
 w_k1_PG_s004_return_constraint: 1.04987 w_k1_PG_s00 - w_k1_PG_s004
                                 + 0.99 b_k1_PG_s004 - 1.01 s_k1_PG_s004  = 0
 w_k1_PG_s005_return_constraint: 1.26609 w_k1_PG_s00 - w_k1_PG_s005
                                 + 0.99 b_k1_PG_s005 - 1.01 s_k1_PG_s005  = 0
 w_k1_PG_s006_return_constraint: 0.740181 w_k1_PG_s00 - w_k1_PG_s006
                                 + 0.99 b_k1_PG_s006 - 1.01 s_k1_PG_s006  = 0
 w_k1_PG_s007_return_constraint: 1.05183 w_k1_PG_s00 - w_k1_PG_s007
                                 + 0.99 b_k1_PG_s007 - 1.01 s_k1_PG_s007  = 0
 w_k1_PG_s010_return_constraint: 0.788894 w_k1_PG_s01 - w_k1_PG_s010
                                 + 0.99 b_k1_PG_s010 - 1.01 s_k1_PG_s010  = 0
 w_k1_PG_s011_return_constraint: 0.694237 w_k1_PG_s01 - w_k1_PG_s011
                                 + 0.99 b_k1_PG_s011 - 1.01 s_k1_PG_s011  = 0
 w_k1_PG_s012_return_constraint: 1.00841 w_k1_PG_s01 - w_k1_PG_s012
                                 + 0.99 b_k1_PG_s012 - 1.01 s_k1_PG_s012  = 0
 w_k1_PG_s013_return_constraint: 1.01466 w_k1_PG_s01 - w_k1_PG_s013
                                 + 0.99 b_k1_PG_s013 - 1.01 s_k1_PG_s013  = 0
 w_k1_PG_s014_return_constraint: 0.90067 w_k1_PG_s01 - w_k1_PG_s014
                                 + 0.99 b_k1_PG_s014 - 1.01 s_k1_PG_s014  = 0
 w_k1_PG_s015_return_constraint: 0.9634 w_k1_PG_s01 - w_k1_PG_s015
                                 + 0.99 b_k1_PG_s015 - 1.01 s_k1_PG_s015  = 0
 w_k1_PG_s016_return_constraint: 1.32265 w_k1_PG_s01 - w_k1_PG_s016
                                 + 0.99 b_k1_PG_s016 - 1.01 s_k1_PG_s016  = 0
 w_k1_PG_s017_return_constraint: 1.26552 w_k1_PG_s01 - w_k1_PG_s017
                                 + 0.99 b_k1_PG_s017 - 1.01 s_k1_PG_s017  = 0
 w_k1_PG_s100_return_constraint: 0.934782 w_k1_PG_s10 - w_k1_PG_s100
                                 + 0.99 b_k1_PG_s100 - 1.01 s_k1_PG_s100  = 0
 w_k1_PG_s101_return_constraint: 0.883776 w_k1_PG_s10 - w_k1_PG_s101
                                 + 0.99 b_k1_PG_s101 - 1.01 s_k1_PG_s101  = 0
 w_k1_PG_s102_return_constraint: 0.862419 w_k1_PG_s10 - w_k1_PG_s102
                                 + 0.99 b_k1_PG_s102 - 1.01 s_k1_PG_s102  = 0
 w_k1_PG_s103_return_constraint: 1.21571 w_k1_PG_s10 - w_k1_PG_s103
                                 + 0.99 b_k1_PG_s103 - 1.01 s_k1_PG_s103  = 0
 w_k1_PG_s104_return_constraint: 1.19417 w_k1_PG_s10 - w_k1_PG_s104
                                 + 0.99 b_k1_PG_s104 - 1.01 s_k1_PG_s104  = 0
 w_k1_PG_s105_return_constraint: 0.881119 w_k1_PG_s10 - w_k1_PG_s105
                                 + 0.99 b_k1_PG_s105 - 1.01 s_k1_PG_s105  = 0
 w_k1_PG_s106_return_constraint: 0.994542 w_k1_PG_s10 - w_k1_PG_s106
                                 + 0.99 b_k1_PG_s106 - 1.01 s_k1_PG_s106  = 0
 w_k1_PG_s107_return_constraint: 0.826537 w_k1_PG_s10 - w_k1_PG_s107
                                 + 0.99 b_k1_PG_s107 - 1.01 s_k1_PG_s107  = 0
 w_k1_PG_s110_return_constraint: 0.934912 w_k1_PG_s11 - w_k1_PG_s110
                                 + 0.99 b_k1_PG_s110 - 1.01 s_k1_PG_s110  = 0
 w_k1_PG_s111_return_constraint: 0.898835 w_k1_PG_s11 - w_k1_PG_s111
                                 + 0.99 b_k1_PG_s111 - 1.01 s_k1_PG_s111  = 0
 w_k1_PG_s112_return_constraint: 1.28913 w_k1_PG_s11 - w_k1_PG_s112
                                 + 0.99 b_k1_PG_s112 - 1.01 s_k1_PG_s112  = 0
 w_k1_PG_s113_return_constraint: 1.16943 w_k1_PG_s11 - w_k1_PG_s113
                                 + 0.99 b_k1_PG_s113 - 1.01 s_k1_PG_s113  = 0
 w_k1_PG_s114_return_constraint: 1.15652 w_k1_PG_s11 - w_k1_PG_s114
                                 + 0.99 b_k1_PG_s114 - 1.01 s_k1_PG_s114  = 0
 w_k1_PG_s115_return_constraint: 0.777392 w_k1_PG_s11 - w_k1_PG_s115
                                 + 0.99 b_k1_PG_s115 - 1.01 s_k1_PG_s115  = 0
 w_k1_PG_s116_return_constraint: 0.95963 w_k1_PG_s11 - w_k1_PG_s116
                                 + 0.99 b_k1_PG_s116 - 1.01 s_k1_PG_s116  = 0
 w_k1_PG_s117_return_constraint: 1.00138 w_k1_PG_s11 - w_k1_PG_s117
                                 + 0.99 b_k1_PG_s117 - 1.01 s_k1_PG_s117  = 0
 k1_s0_balance_constraint:       b_k1_ADM_s0 + b_k1_AXP_s0 + b_k1_DIS_s0
                                 + b_k1_FDX_s0 + b_k1_IBM_s0 + b_k1_INTC_s0
                                 + b_k1_KO_s0 + b_k1_MAT_s0 + b_k1_MSFT_s0
                                 + b_k1_PG_s0 - s_k1_ADM_s0 - s_k1_AXP_s0
                                 - s_k1_DIS_s0 - s_k1_FDX_s0 - s_k1_IBM_s0
                                 - s_k1_INTC_s0 - s_k1_KO_s0 - s_k1_MAT_s0
                                 - s_k1_MSFT_s0 - s_k1_PG_s0  = 0
 k1_s1_balance_constraint:       b_k1_ADM_s1 + b_k1_AXP_s1 + b_k1_DIS_s1
                                 + b_k1_FDX_s1 + b_k1_IBM_s1 + b_k1_INTC_s1
                                 + b_k1_KO_s1 + b_k1_MAT_s1 + b_k1_MSFT_s1
                                 + b_k1_PG_s1 - s_k1_ADM_s1 - s_k1_AXP_s1
                                 - s_k1_DIS_s1 - s_k1_FDX_s1 - s_k1_IBM_s1
                                 - s_k1_INTC_s1 - s_k1_KO_s1 - s_k1_MAT_s1
                                 - s_k1_MSFT_s1 - s_k1_PG_s1  = 0
 k1_s00_balance_constraint:      b_k1_ADM_s00 + b_k1_AXP_s00 + b_k1_DIS_s00
                                 + b_k1_FDX_s00 + b_k1_IBM_s00 + b_k1_INTC_s00
                                 + b_k1_KO_s00 + b_k1_MAT_s00 + b_k1_MSFT_s00
                                 + b_k1_PG_s00 - s_k1_ADM_s00 - s_k1_AXP_s00
                                 - s_k1_DIS_s00 - s_k1_FDX_s00 - s_k1_IBM_s00
                                 - s_k1_INTC_s00 - s_k1_KO_s00 - s_k1_MAT_s00
                                 - s_k1_MSFT_s00 - s_k1_PG_s00  = 0
 k1_s01_balance_constraint:      b_k1_ADM_s01 + b_k1_AXP_s01 + b_k1_DIS_s01
                                 + b_k1_FDX_s01 + b_k1_IBM_s01 + b_k1_INTC_s01
                                 + b_k1_KO_s01 + b_k1_MAT_s01 + b_k1_MSFT_s01
                                 + b_k1_PG_s01 - s_k1_ADM_s01 - s_k1_AXP_s01
                                 - s_k1_DIS_s01 - s_k1_FDX_s01 - s_k1_IBM_s01
                                 - s_k1_INTC_s01 - s_k1_KO_s01 - s_k1_MAT_s01
                                 - s_k1_MSFT_s01 - s_k1_PG_s01  = 0
 k1_s10_balance_constraint:      b_k1_ADM_s10 + b_k1_AXP_s10 + b_k1_DIS_s10
                                 + b_k1_FDX_s10 + b_k1_IBM_s10 + b_k1_INTC_s10
                                 + b_k1_KO_s10 + b_k1_MAT_s10 + b_k1_MSFT_s10
                                 + b_k1_PG_s10 - s_k1_ADM_s10 - s_k1_AXP_s10
                                 - s_k1_DIS_s10 - s_k1_FDX_s10 - s_k1_IBM_s10
                                 - s_k1_INTC_s10 - s_k1_KO_s10 - s_k1_MAT_s10
                                 - s_k1_MSFT_s10 - s_k1_PG_s10  = 0
 k1_s11_balance_constraint:      b_k1_ADM_s11 + b_k1_AXP_s11 + b_k1_DIS_s11
                                 + b_k1_FDX_s11 + b_k1_IBM_s11 + b_k1_INTC_s11
                                 + b_k1_KO_s11 + b_k1_MAT_s11 + b_k1_MSFT_s11
                                 + b_k1_PG_s11 - s_k1_ADM_s11 - s_k1_AXP_s11
                                 - s_k1_DIS_s11 - s_k1_FDX_s11 - s_k1_IBM_s11
                                 - s_k1_INTC_s11 - s_k1_KO_s11 - s_k1_MAT_s11
                                 - s_k1_MSFT_s11 - s_k1_PG_s11  = 0
 k1_s000_balance_constraint:     b_k1_ADM_s000 + b_k1_AXP_s000 + b_k1_DIS_s000
                                 + b_k1_FDX_s000 + b_k1_IBM_s000
                                 + b_k1_INTC_s000 + b_k1_KO_s000
                                 + b_k1_MAT_s000 + b_k1_MSFT_s000
                                 + b_k1_PG_s000 - s_k1_ADM_s000 - s_k1_AXP_s000
                                 - s_k1_DIS_s000 - s_k1_FDX_s000
                                 - s_k1_IBM_s000 - s_k1_INTC_s000
                                 - s_k1_KO_s000 - s_k1_MAT_s000
                                 - s_k1_MSFT_s000 - s_k1_PG_s000  = 0
 k1_s001_balance_constraint:     b_k1_ADM_s001 + b_k1_AXP_s001 + b_k1_DIS_s001
                                 + b_k1_FDX_s001 + b_k1_IBM_s001
                                 + b_k1_INTC_s001 + b_k1_KO_s001
                                 + b_k1_MAT_s001 + b_k1_MSFT_s001
                                 + b_k1_PG_s001 - s_k1_ADM_s001 - s_k1_AXP_s001
                                 - s_k1_DIS_s001 - s_k1_FDX_s001
                                 - s_k1_IBM_s001 - s_k1_INTC_s001
                                 - s_k1_KO_s001 - s_k1_MAT_s001
                                 - s_k1_MSFT_s001 - s_k1_PG_s001  = 0
 k1_s002_balance_constraint:     b_k1_ADM_s002 + b_k1_AXP_s002 + b_k1_DIS_s002
                                 + b_k1_FDX_s002 + b_k1_IBM_s002
                                 + b_k1_INTC_s002 + b_k1_KO_s002
                                 + b_k1_MAT_s002 + b_k1_MSFT_s002
                                 + b_k1_PG_s002 - s_k1_ADM_s002 - s_k1_AXP_s002
                                 - s_k1_DIS_s002 - s_k1_FDX_s002
                                 - s_k1_IBM_s002 - s_k1_INTC_s002
                                 - s_k1_KO_s002 - s_k1_MAT_s002
                                 - s_k1_MSFT_s002 - s_k1_PG_s002  = 0
 k1_s003_balance_constraint:     b_k1_ADM_s003 + b_k1_AXP_s003 + b_k1_DIS_s003
                                 + b_k1_FDX_s003 + b_k1_IBM_s003
                                 + b_k1_INTC_s003 + b_k1_KO_s003
                                 + b_k1_MAT_s003 + b_k1_MSFT_s003
                                 + b_k1_PG_s003 - s_k1_ADM_s003 - s_k1_AXP_s003
                                 - s_k1_DIS_s003 - s_k1_FDX_s003
                                 - s_k1_IBM_s003 - s_k1_INTC_s003
                                 - s_k1_KO_s003 - s_k1_MAT_s003
                                 - s_k1_MSFT_s003 - s_k1_PG_s003  = 0
 k1_s004_balance_constraint:     b_k1_ADM_s004 + b_k1_AXP_s004 + b_k1_DIS_s004
                                 + b_k1_FDX_s004 + b_k1_IBM_s004
                                 + b_k1_INTC_s004 + b_k1_KO_s004
                                 + b_k1_MAT_s004 + b_k1_MSFT_s004
                                 + b_k1_PG_s004 - s_k1_ADM_s004 - s_k1_AXP_s004
                                 - s_k1_DIS_s004 - s_k1_FDX_s004
                                 - s_k1_IBM_s004 - s_k1_INTC_s004
                                 - s_k1_KO_s004 - s_k1_MAT_s004
                                 - s_k1_MSFT_s004 - s_k1_PG_s004  = 0
 k1_s005_balance_constraint:     b_k1_ADM_s005 + b_k1_AXP_s005 + b_k1_DIS_s005
                                 + b_k1_FDX_s005 + b_k1_IBM_s005
                                 + b_k1_INTC_s005 + b_k1_KO_s005
                                 + b_k1_MAT_s005 + b_k1_MSFT_s005
                                 + b_k1_PG_s005 - s_k1_ADM_s005 - s_k1_AXP_s005
                                 - s_k1_DIS_s005 - s_k1_FDX_s005
                                 - s_k1_IBM_s005 - s_k1_INTC_s005
                                 - s_k1_KO_s005 - s_k1_MAT_s005
                                 - s_k1_MSFT_s005 - s_k1_PG_s005  = 0
 k1_s006_balance_constraint:     b_k1_ADM_s006 + b_k1_AXP_s006 + b_k1_DIS_s006
                                 + b_k1_FDX_s006 + b_k1_IBM_s006
                                 + b_k1_INTC_s006 + b_k1_KO_s006
                                 + b_k1_MAT_s006 + b_k1_MSFT_s006
                                 + b_k1_PG_s006 - s_k1_ADM_s006 - s_k1_AXP_s006
                                 - s_k1_DIS_s006 - s_k1_FDX_s006
                                 - s_k1_IBM_s006 - s_k1_INTC_s006
                                 - s_k1_KO_s006 - s_k1_MAT_s006
                                 - s_k1_MSFT_s006 - s_k1_PG_s006  = 0
 k1_s007_balance_constraint:     b_k1_ADM_s007 + b_k1_AXP_s007 + b_k1_DIS_s007
                                 + b_k1_FDX_s007 + b_k1_IBM_s007
                                 + b_k1_INTC_s007 + b_k1_KO_s007
                                 + b_k1_MAT_s007 + b_k1_MSFT_s007
                                 + b_k1_PG_s007 - s_k1_ADM_s007 - s_k1_AXP_s007
                                 - s_k1_DIS_s007 - s_k1_FDX_s007
                                 - s_k1_IBM_s007 - s_k1_INTC_s007
                                 - s_k1_KO_s007 - s_k1_MAT_s007
                                 - s_k1_MSFT_s007 - s_k1_PG_s007  = 0
 k1_s010_balance_constraint:     b_k1_ADM_s010 + b_k1_AXP_s010 + b_k1_DIS_s010
                                 + b_k1_FDX_s010 + b_k1_IBM_s010
                                 + b_k1_INTC_s010 + b_k1_KO_s010
                                 + b_k1_MAT_s010 + b_k1_MSFT_s010
                                 + b_k1_PG_s010 - s_k1_ADM_s010 - s_k1_AXP_s010
                                 - s_k1_DIS_s010 - s_k1_FDX_s010
                                 - s_k1_IBM_s010 - s_k1_INTC_s010
                                 - s_k1_KO_s010 - s_k1_MAT_s010
                                 - s_k1_MSFT_s010 - s_k1_PG_s010  = 0
 k1_s011_balance_constraint:     b_k1_ADM_s011 + b_k1_AXP_s011 + b_k1_DIS_s011
                                 + b_k1_FDX_s011 + b_k1_IBM_s011
                                 + b_k1_INTC_s011 + b_k1_KO_s011
                                 + b_k1_MAT_s011 + b_k1_MSFT_s011
                                 + b_k1_PG_s011 - s_k1_ADM_s011 - s_k1_AXP_s011
                                 - s_k1_DIS_s011 - s_k1_FDX_s011
                                 - s_k1_IBM_s011 - s_k1_INTC_s011
                                 - s_k1_KO_s011 - s_k1_MAT_s011
                                 - s_k1_MSFT_s011 - s_k1_PG_s011  = 0
 k1_s012_balance_constraint:     b_k1_ADM_s012 + b_k1_AXP_s012 + b_k1_DIS_s012
                                 + b_k1_FDX_s012 + b_k1_IBM_s012
                                 + b_k1_INTC_s012 + b_k1_KO_s012
                                 + b_k1_MAT_s012 + b_k1_MSFT_s012
                                 + b_k1_PG_s012 - s_k1_ADM_s012 - s_k1_AXP_s012
                                 - s_k1_DIS_s012 - s_k1_FDX_s012
                                 - s_k1_IBM_s012 - s_k1_INTC_s012
                                 - s_k1_KO_s012 - s_k1_MAT_s012
                                 - s_k1_MSFT_s012 - s_k1_PG_s012  = 0
 k1_s013_balance_constraint:     b_k1_ADM_s013 + b_k1_AXP_s013 + b_k1_DIS_s013
                                 + b_k1_FDX_s013 + b_k1_IBM_s013
                                 + b_k1_INTC_s013 + b_k1_KO_s013
                                 + b_k1_MAT_s013 + b_k1_MSFT_s013
                                 + b_k1_PG_s013 - s_k1_ADM_s013 - s_k1_AXP_s013
                                 - s_k1_DIS_s013 - s_k1_FDX_s013
                                 - s_k1_IBM_s013 - s_k1_INTC_s013
                                 - s_k1_KO_s013 - s_k1_MAT_s013
                                 - s_k1_MSFT_s013 - s_k1_PG_s013  = 0
 k1_s014_balance_constraint:     b_k1_ADM_s014 + b_k1_AXP_s014 + b_k1_DIS_s014
                                 + b_k1_FDX_s014 + b_k1_IBM_s014
                                 + b_k1_INTC_s014 + b_k1_KO_s014
                                 + b_k1_MAT_s014 + b_k1_MSFT_s014
                                 + b_k1_PG_s014 - s_k1_ADM_s014 - s_k1_AXP_s014
                                 - s_k1_DIS_s014 - s_k1_FDX_s014
                                 - s_k1_IBM_s014 - s_k1_INTC_s014
                                 - s_k1_KO_s014 - s_k1_MAT_s014
                                 - s_k1_MSFT_s014 - s_k1_PG_s014  = 0
 k1_s015_balance_constraint:     b_k1_ADM_s015 + b_k1_AXP_s015 + b_k1_DIS_s015
                                 + b_k1_FDX_s015 + b_k1_IBM_s015
                                 + b_k1_INTC_s015 + b_k1_KO_s015
                                 + b_k1_MAT_s015 + b_k1_MSFT_s015
                                 + b_k1_PG_s015 - s_k1_ADM_s015 - s_k1_AXP_s015
                                 - s_k1_DIS_s015 - s_k1_FDX_s015
                                 - s_k1_IBM_s015 - s_k1_INTC_s015
                                 - s_k1_KO_s015 - s_k1_MAT_s015
                                 - s_k1_MSFT_s015 - s_k1_PG_s015  = 0
 k1_s016_balance_constraint:     b_k1_ADM_s016 + b_k1_AXP_s016 + b_k1_DIS_s016
                                 + b_k1_FDX_s016 + b_k1_IBM_s016
                                 + b_k1_INTC_s016 + b_k1_KO_s016
                                 + b_k1_MAT_s016 + b_k1_MSFT_s016
                                 + b_k1_PG_s016 - s_k1_ADM_s016 - s_k1_AXP_s016
                                 - s_k1_DIS_s016 - s_k1_FDX_s016
                                 - s_k1_IBM_s016 - s_k1_INTC_s016
                                 - s_k1_KO_s016 - s_k1_MAT_s016
                                 - s_k1_MSFT_s016 - s_k1_PG_s016  = 0
 k1_s017_balance_constraint:     b_k1_ADM_s017 + b_k1_AXP_s017 + b_k1_DIS_s017
                                 + b_k1_FDX_s017 + b_k1_IBM_s017
                                 + b_k1_INTC_s017 + b_k1_KO_s017
                                 + b_k1_MAT_s017 + b_k1_MSFT_s017
                                 + b_k1_PG_s017 - s_k1_ADM_s017 - s_k1_AXP_s017
                                 - s_k1_DIS_s017 - s_k1_FDX_s017
                                 - s_k1_IBM_s017 - s_k1_INTC_s017
                                 - s_k1_KO_s017 - s_k1_MAT_s017
                                 - s_k1_MSFT_s017 - s_k1_PG_s017  = 0
 k1_s100_balance_constraint:     b_k1_ADM_s100 + b_k1_AXP_s100 + b_k1_DIS_s100
                                 + b_k1_FDX_s100 + b_k1_IBM_s100
                                 + b_k1_INTC_s100 + b_k1_KO_s100
                                 + b_k1_MAT_s100 + b_k1_MSFT_s100
                                 + b_k1_PG_s100 - s_k1_ADM_s100 - s_k1_AXP_s100
                                 - s_k1_DIS_s100 - s_k1_FDX_s100
                                 - s_k1_IBM_s100 - s_k1_INTC_s100
                                 - s_k1_KO_s100 - s_k1_MAT_s100
                                 - s_k1_MSFT_s100 - s_k1_PG_s100  = 0
 k1_s101_balance_constraint:     b_k1_ADM_s101 + b_k1_AXP_s101 + b_k1_DIS_s101
                                 + b_k1_FDX_s101 + b_k1_IBM_s101
                                 + b_k1_INTC_s101 + b_k1_KO_s101
                                 + b_k1_MAT_s101 + b_k1_MSFT_s101
                                 + b_k1_PG_s101 - s_k1_ADM_s101 - s_k1_AXP_s101
                                 - s_k1_DIS_s101 - s_k1_FDX_s101
                                 - s_k1_IBM_s101 - s_k1_INTC_s101
                                 - s_k1_KO_s101 - s_k1_MAT_s101
                                 - s_k1_MSFT_s101 - s_k1_PG_s101  = 0
 k1_s102_balance_constraint:     b_k1_ADM_s102 + b_k1_AXP_s102 + b_k1_DIS_s102
                                 + b_k1_FDX_s102 + b_k1_IBM_s102
                                 + b_k1_INTC_s102 + b_k1_KO_s102
                                 + b_k1_MAT_s102 + b_k1_MSFT_s102
                                 + b_k1_PG_s102 - s_k1_ADM_s102 - s_k1_AXP_s102
                                 - s_k1_DIS_s102 - s_k1_FDX_s102
                                 - s_k1_IBM_s102 - s_k1_INTC_s102
                                 - s_k1_KO_s102 - s_k1_MAT_s102
                                 - s_k1_MSFT_s102 - s_k1_PG_s102  = 0
 k1_s103_balance_constraint:     b_k1_ADM_s103 + b_k1_AXP_s103 + b_k1_DIS_s103
                                 + b_k1_FDX_s103 + b_k1_IBM_s103
                                 + b_k1_INTC_s103 + b_k1_KO_s103
                                 + b_k1_MAT_s103 + b_k1_MSFT_s103
                                 + b_k1_PG_s103 - s_k1_ADM_s103 - s_k1_AXP_s103
                                 - s_k1_DIS_s103 - s_k1_FDX_s103
                                 - s_k1_IBM_s103 - s_k1_INTC_s103
                                 - s_k1_KO_s103 - s_k1_MAT_s103
                                 - s_k1_MSFT_s103 - s_k1_PG_s103  = 0
 k1_s104_balance_constraint:     b_k1_ADM_s104 + b_k1_AXP_s104 + b_k1_DIS_s104
                                 + b_k1_FDX_s104 + b_k1_IBM_s104
                                 + b_k1_INTC_s104 + b_k1_KO_s104
                                 + b_k1_MAT_s104 + b_k1_MSFT_s104
                                 + b_k1_PG_s104 - s_k1_ADM_s104 - s_k1_AXP_s104
                                 - s_k1_DIS_s104 - s_k1_FDX_s104
                                 - s_k1_IBM_s104 - s_k1_INTC_s104
                                 - s_k1_KO_s104 - s_k1_MAT_s104
                                 - s_k1_MSFT_s104 - s_k1_PG_s104  = 0
 k1_s105_balance_constraint:     b_k1_ADM_s105 + b_k1_AXP_s105 + b_k1_DIS_s105
                                 + b_k1_FDX_s105 + b_k1_IBM_s105
                                 + b_k1_INTC_s105 + b_k1_KO_s105
                                 + b_k1_MAT_s105 + b_k1_MSFT_s105
                                 + b_k1_PG_s105 - s_k1_ADM_s105 - s_k1_AXP_s105
                                 - s_k1_DIS_s105 - s_k1_FDX_s105
                                 - s_k1_IBM_s105 - s_k1_INTC_s105
                                 - s_k1_KO_s105 - s_k1_MAT_s105
                                 - s_k1_MSFT_s105 - s_k1_PG_s105  = 0
 k1_s106_balance_constraint:     b_k1_ADM_s106 + b_k1_AXP_s106 + b_k1_DIS_s106
                                 + b_k1_FDX_s106 + b_k1_IBM_s106
                                 + b_k1_INTC_s106 + b_k1_KO_s106
                                 + b_k1_MAT_s106 + b_k1_MSFT_s106
                                 + b_k1_PG_s106 - s_k1_ADM_s106 - s_k1_AXP_s106
                                 - s_k1_DIS_s106 - s_k1_FDX_s106
                                 - s_k1_IBM_s106 - s_k1_INTC_s106
                                 - s_k1_KO_s106 - s_k1_MAT_s106
                                 - s_k1_MSFT_s106 - s_k1_PG_s106  = 0
 k1_s107_balance_constraint:     b_k1_ADM_s107 + b_k1_AXP_s107 + b_k1_DIS_s107
                                 + b_k1_FDX_s107 + b_k1_IBM_s107
                                 + b_k1_INTC_s107 + b_k1_KO_s107
                                 + b_k1_MAT_s107 + b_k1_MSFT_s107
                                 + b_k1_PG_s107 - s_k1_ADM_s107 - s_k1_AXP_s107
                                 - s_k1_DIS_s107 - s_k1_FDX_s107
                                 - s_k1_IBM_s107 - s_k1_INTC_s107
                                 - s_k1_KO_s107 - s_k1_MAT_s107
                                 - s_k1_MSFT_s107 - s_k1_PG_s107  = 0
 k1_s110_balance_constraint:     b_k1_ADM_s110 + b_k1_AXP_s110 + b_k1_DIS_s110
                                 + b_k1_FDX_s110 + b_k1_IBM_s110
                                 + b_k1_INTC_s110 + b_k1_KO_s110
                                 + b_k1_MAT_s110 + b_k1_MSFT_s110
                                 + b_k1_PG_s110 - s_k1_ADM_s110 - s_k1_AXP_s110
                                 - s_k1_DIS_s110 - s_k1_FDX_s110
                                 - s_k1_IBM_s110 - s_k1_INTC_s110
                                 - s_k1_KO_s110 - s_k1_MAT_s110
                                 - s_k1_MSFT_s110 - s_k1_PG_s110  = 0
 k1_s111_balance_constraint:     b_k1_ADM_s111 + b_k1_AXP_s111 + b_k1_DIS_s111
                                 + b_k1_FDX_s111 + b_k1_IBM_s111
                                 + b_k1_INTC_s111 + b_k1_KO_s111
                                 + b_k1_MAT_s111 + b_k1_MSFT_s111
                                 + b_k1_PG_s111 - s_k1_ADM_s111 - s_k1_AXP_s111
                                 - s_k1_DIS_s111 - s_k1_FDX_s111
                                 - s_k1_IBM_s111 - s_k1_INTC_s111
                                 - s_k1_KO_s111 - s_k1_MAT_s111
                                 - s_k1_MSFT_s111 - s_k1_PG_s111  = 0
 k1_s112_balance_constraint:     b_k1_ADM_s112 + b_k1_AXP_s112 + b_k1_DIS_s112
                                 + b_k1_FDX_s112 + b_k1_IBM_s112
                                 + b_k1_INTC_s112 + b_k1_KO_s112
                                 + b_k1_MAT_s112 + b_k1_MSFT_s112
                                 + b_k1_PG_s112 - s_k1_ADM_s112 - s_k1_AXP_s112
                                 - s_k1_DIS_s112 - s_k1_FDX_s112
                                 - s_k1_IBM_s112 - s_k1_INTC_s112
                                 - s_k1_KO_s112 - s_k1_MAT_s112
                                 - s_k1_MSFT_s112 - s_k1_PG_s112  = 0
 k1_s113_balance_constraint:     b_k1_ADM_s113 + b_k1_AXP_s113 + b_k1_DIS_s113
                                 + b_k1_FDX_s113 + b_k1_IBM_s113
                                 + b_k1_INTC_s113 + b_k1_KO_s113
                                 + b_k1_MAT_s113 + b_k1_MSFT_s113
                                 + b_k1_PG_s113 - s_k1_ADM_s113 - s_k1_AXP_s113
                                 - s_k1_DIS_s113 - s_k1_FDX_s113
                                 - s_k1_IBM_s113 - s_k1_INTC_s113
                                 - s_k1_KO_s113 - s_k1_MAT_s113
                                 - s_k1_MSFT_s113 - s_k1_PG_s113  = 0
 k1_s114_balance_constraint:     b_k1_ADM_s114 + b_k1_AXP_s114 + b_k1_DIS_s114
                                 + b_k1_FDX_s114 + b_k1_IBM_s114
                                 + b_k1_INTC_s114 + b_k1_KO_s114
                                 + b_k1_MAT_s114 + b_k1_MSFT_s114
                                 + b_k1_PG_s114 - s_k1_ADM_s114 - s_k1_AXP_s114
                                 - s_k1_DIS_s114 - s_k1_FDX_s114
                                 - s_k1_IBM_s114 - s_k1_INTC_s114
                                 - s_k1_KO_s114 - s_k1_MAT_s114
                                 - s_k1_MSFT_s114 - s_k1_PG_s114  = 0
 k1_s115_balance_constraint:     b_k1_ADM_s115 + b_k1_AXP_s115 + b_k1_DIS_s115
                                 + b_k1_FDX_s115 + b_k1_IBM_s115
                                 + b_k1_INTC_s115 + b_k1_KO_s115
                                 + b_k1_MAT_s115 + b_k1_MSFT_s115
                                 + b_k1_PG_s115 - s_k1_ADM_s115 - s_k1_AXP_s115
                                 - s_k1_DIS_s115 - s_k1_FDX_s115
                                 - s_k1_IBM_s115 - s_k1_INTC_s115
                                 - s_k1_KO_s115 - s_k1_MAT_s115
                                 - s_k1_MSFT_s115 - s_k1_PG_s115  = 0
 k1_s116_balance_constraint:     b_k1_ADM_s116 + b_k1_AXP_s116 + b_k1_DIS_s116
                                 + b_k1_FDX_s116 + b_k1_IBM_s116
                                 + b_k1_INTC_s116 + b_k1_KO_s116
                                 + b_k1_MAT_s116 + b_k1_MSFT_s116
                                 + b_k1_PG_s116 - s_k1_ADM_s116 - s_k1_AXP_s116
                                 - s_k1_DIS_s116 - s_k1_FDX_s116
                                 - s_k1_IBM_s116 - s_k1_INTC_s116
                                 - s_k1_KO_s116 - s_k1_MAT_s116
                                 - s_k1_MSFT_s116 - s_k1_PG_s116  = 0
 k1_s117_balance_constraint:     b_k1_ADM_s117 + b_k1_AXP_s117 + b_k1_DIS_s117
                                 + b_k1_FDX_s117 + b_k1_IBM_s117
                                 + b_k1_INTC_s117 + b_k1_KO_s117
                                 + b_k1_MAT_s117 + b_k1_MSFT_s117
                                 + b_k1_PG_s117 - s_k1_ADM_s117 - s_k1_AXP_s117
                                 - s_k1_DIS_s117 - s_k1_FDX_s117
                                 - s_k1_IBM_s117 - s_k1_INTC_s117
                                 - s_k1_KO_s117 - s_k1_MAT_s117
                                 - s_k1_MSFT_s117 - s_k1_PG_s117  = 0
 k1_wcvar_constraint:            wcvar - var_1 - 0.860001824360604 z_k1_s0000
                                 - 0.748003924390814 z_k1_s0001
                                 - 0.352000485270544 z_k1_s0002
                                 - 1.02799867431529 z_k1_s0003
                                 - 0.5900006518118 z_k1_s0004
                                 - 1.03600041374393 z_k1_s0005
                                 - 1.10999997093937 z_k1_s0006
                                 - 0.888999863529683 z_k1_s0007
                                 - 0.19200003248431 z_k1_s0010
                                 - 0.221000080565813 z_k1_s0011
                                 - 0.468000879880861 z_k1_s0012
                                 - 0.141000129830712 z_k1_s0013
                                 - 0.400001009676064 z_k1_s0014
                                 - 0.684000319825444 z_k1_s0015
                                 - 0.141000129830712 z_k1_s0016
                                 - 0.26500104571056 z_k1_s0017
                                 - 0.578997700857604 z_k1_s0020
                                 - 0.266999196340234 z_k1_s0021
                                 - 0.324999025599523 z_k1_s0022
                                 - 0.16299946716852 z_k1_s0023
                                 - 0.651996502678538 z_k1_s0024
                                 - 0.51799916988242 z_k1_s0025
                                 - 0.308999119803704 z_k1_s0026
                                 - 0.601997650813816 z_k1_s0027
                                 - 0.189000205064343 z_k1_s0030
                                 - 0.646999091080356 z_k1_s0031
                                 - 0.430001760327312 z_k1_s0032
                                 - 0.532001151529718 z_k1_s0033
                                 - 0.324000521939009 z_k1_s0034
                                 - 0.191000234056579 z_k1_s0035
                                 - 0.440000414287308 z_k1_s0036
                                 - 0.2290001885086 z_k1_s0037
                                 - 0.162000184298694 z_k1_s0040
                                 - 0.393999142831923 z_k1_s0041
                                 - 0.268000251389168 z_k1_s0042
                                 - 0.541000208866406 z_k1_s0043
                                 - 0.477999751994485 z_k1_s0044
                                 - 0.505000360511244 z_k1_s0045
                                 - 0.370000206929 z_k1_s0046
                                 - 0.171000001937406 z_k1_s0047
                                 - 0.417000176332476 z_k1_s0050
                                 - 0.135000043119856 z_k1_s0051
                                 - 0.242000479936632 z_k1_s0052
                                 - 0.213000240593678 z_k1_s0053
                                 - 0.142000160395785 z_k1_s0054
                                 - 0.342999922616236 z_k1_s0055
                                 - 0.431999965423411 z_k1_s0056
                                 - 0.232999312281114 z_k1_s0057
                                 - 0.23999940786549 z_k1_s0060
                                 - 0.353999244641297 z_k1_s0061
                                 - 0.631999082364159 z_k1_s0062
                                 - 0.648996799180325 z_k1_s0063
                                 - 0.848999635059775 z_k1_s0064
                                 - 0.234999246168734 z_k1_s0065
                                 - 0.387998310264402 z_k1_s0066
                                 - 0.28499941033998 z_k1_s0067
                                 - 1.19000355020498 z_k1_s0070
                                 - 0.944999381628982 z_k1_s0071
                                 - 0.526000410270639 z_k1_s0072
                                 - 1.13799967585851 z_k1_s0073
                                 - 0.421000738712195 z_k1_s0074
                                 - 0.711000647893702 z_k1_s0075
                                 - 1.24400217909636 z_k1_s0076
                                 - 1.61700569497376 z_k1_s0077
                                 - 0.310999442923441 z_k1_s0100
                                 - 0.114999889911145 z_k1_s0101
                                 - 0.087999883144213 z_k1_s0102
                                 - 0.123999892166789 z_k1_s0103
                                 - 0.295998843831293 z_k1_s0104
                                 - 0.198999494230903 z_k1_s0105
                                 - 0.218999102354951 z_k1_s0106
                                 - 0.433999354684028 z_k1_s0107
                                 - 0.459001921248494 z_k1_s0110
                                 - 0.495002014699251 z_k1_s0111
                                 - 0.465000477320926 z_k1_s0112
                                 - 0.327999878438431 z_k1_s0113
                                 - 0.227000540608848 z_k1_s0114
                                 - 0.167000384857587 z_k1_s0115
                                 - 0.281000388884444 z_k1_s0116
                                 - 0.497001533390062 z_k1_s0117
                                 - 0.562999703717232 z_k1_s0120
                                 - 0.778997976196775 z_k1_s0121
                                 - 0.192999205944474 z_k1_s0122
                                 - 0.911995918236138 z_k1_s0123
                                 - 0.647998947424717 z_k1_s0124
                                 - 0.303998777078249 z_k1_s0125
                                 - 0.509999592230999 z_k1_s0126
                                 - 0.220999272635288 z_k1_s0127
                                 - 0.478998178310773 z_k1_s0130
                                 - 0.270999327486301 z_k1_s0131
                                 - 0.219000660778467 z_k1_s0132
                                 - 0.281999045443727 z_k1_s0133
                                 - 0.104999818171316 z_k1_s0134
                                 - 0.342999573386024 z_k1_s0135
                                 - 0.149999710359072 z_k1_s0136
                                 - 0.243000045412852 z_k1_s0137
                                 - 0.541000114711852 z_k1_s0140
                                 - 0.250999905051308 z_k1_s0141
                                 - 1.11199757335688 z_k1_s0142
                                 - 0.473999784100915 z_k1_s0143
                                 - 0.237999803514897 z_k1_s0144
                                 - 0.995999493490585 z_k1_s0145
                                 - 0.298999830340063 z_k1_s0146
                                 - 0.263999589088152 z_k1_s0147
                                 - 0.56700157890873 z_k1_s0150
                                 - 1.02099825140622 z_k1_s0151
                                 - 0.330999950285554 z_k1_s0152
                                 - 0.466001333529905 z_k1_s0153
                                 - 0.401999933476719 z_k1_s0154
                                 - 0.334999605564129 z_k1_s0155
                                 - 0.572997807428481 z_k1_s0156
                                 - 0.372999585108703 z_k1_s0157
                                 - 0.176000040813927 z_k1_s0160
                                 - 0.272999498637863 z_k1_s0161
                                 - 0.103999975280987 z_k1_s0162
                                 - 0.0759999199515292 z_k1_s0163
                                 - 0.0829999002089146 z_k1_s0164
                                 - 0.184000286850773 z_k1_s0165
                                 - 0.096999995023602 z_k1_s0166
                                 - 0.350000355868435 z_k1_s0167
                                 - 0.134999773800018 z_k1_s0170
                                 - 0.188999647940083 z_k1_s0171
                                 - 0.216999333709417 z_k1_s0172
                                 - 0.178999507451453 z_k1_s0173
                                 - 0.333999031529608 z_k1_s0174
                                 - 0.122999782113369 z_k1_s0175
                                 - 0.43800013881195 z_k1_s0176
                                 - 0.153999686929 z_k1_s0177
                                 - 0.743000511402892 z_k1_s1000
                                 - 0.663997264196105 z_k1_s1001
                                 - 0.695000053630284 z_k1_s1002
                                 - 0.87799703801774 z_k1_s1003
                                 - 0.353999839426246 z_k1_s1004
                                 - 1.03499694835878 z_k1_s1005
                                 - 0.207999988649808 z_k1_s1006
                                 - 0.863997357916978 z_k1_s1007
                                 - 0.317000594235826 z_k1_s1010
                                 - 0.24100061987973 z_k1_s1011
                                 - 0.541001372286018 z_k1_s1012
                                 - 0.202000151346632 z_k1_s1013
                                 - 0.150000144502969 z_k1_s1014
                                 - 0.516000923418536 z_k1_s1015
                                 - 0.11400019323432 z_k1_s1016
                                 - 0.236000530106233 z_k1_s1017
                                 - 0.0880000260744662 z_k1_s1020
                                 - 0.221999863701493 z_k1_s1021
                                 - 0.237000268816461 z_k1_s1022
                                 - 0.303000518322313 z_k1_s1023
                                 - 0.0839999691438082 z_k1_s1024
                                 - 0.120000021619726 z_k1_s1025
                                 - 0.367000202812831 z_k1_s1026
                                 - 0.112000061058412 z_k1_s1027
                                 - 0.570000363538729 z_k1_s1030
                                 - 1.23500168883365 z_k1_s1031
                                 - 0.685001812348766 z_k1_s1032
                                 - 0.411999760012354 z_k1_s1033
                                 - 1.01100055942925 z_k1_s1034
                                 - 0.415999857423124 z_k1_s1035
                                 - 0.418999660131281 z_k1_s1036
                                 - 0.658999557079244 z_k1_s1037
                                 - 0.523001177248927 z_k1_s1040
                                 - 0.310000947480366 z_k1_s1041
                                 - 0.541999971593877 z_k1_s1042
                                 - 0.119999945524451 z_k1_s1043
                                 - 0.34100024272801 z_k1_s1044
                                 - 0.306000780512802 z_k1_s1045
                                 - 0.171999984101743 z_k1_s1046
                                 - 0.350999327646266 z_k1_s1047
                                 - 0.0579999627759999 z_k1_s1050
                                 - 0.121000094391387 z_k1_s1051
                                 - 0.14699960683559 z_k1_s1052
                                 - 0.28499990764073 z_k1_s1053
                                 - 0.154000525971038 z_k1_s1054
                                 - 0.141000507148819 z_k1_s1055
                                 - 0.14199970059632 z_k1_s1056
                                 - 0.26500028268365 z_k1_s1057
                                 - 0.329999908833403 z_k1_s1060
                                 - 0.268999758721966 z_k1_s1061
                                 - 0.189999724892215 z_k1_s1062
                                 - 0.329999908833403 z_k1_s1063
                                 - 0.227999853530447 z_k1_s1064
                                 - 0.806000166913865 z_k1_s1065
                                 - 0.405000444658537 z_k1_s1066
                                 - 0.502998247060042 z_k1_s1067
                                 - 0.107000120011368 z_k1_s1070
                                 - 0.11100015554265 z_k1_s1071
                                 - 0.403000990725345 z_k1_s1072
                                 - 0.393000413396878 z_k1_s1073
                                 - 0.184000071238168 z_k1_s1074
                                 - 0.128000160000523 z_k1_s1075
                                 - 0.285000821852976 z_k1_s1076
                                 - 0.342999480755585 z_k1_s1077
                                 - 0.391001339533808 z_k1_s1100
                                 - 0.578999843458873 z_k1_s1101
                                 - 0.241000473381011 z_k1_s1102
                                 - 0.222000585592762 z_k1_s1103
                                 - 0.439001152493784 z_k1_s1104
                                 - 0.585000965331135 z_k1_s1105
                                 - 0.380999469746703 z_k1_s1106
                                 - 0.216000379922312 z_k1_s1107
                                 - 0.540999728164472 z_k1_s1110
                                 - 0.122000203534507 z_k1_s1111
                                 - 0.425999585652472 z_k1_s1112
                                 - 0.227000215871466 z_k1_s1113
                                 - 0.42699890636948 z_k1_s1114
                                 - 0.474000090092583 z_k1_s1115
                                 - 0.536000114577276 z_k1_s1116
                                 - 0.257000305396373 z_k1_s1117
                                 - 0.277999973133703 z_k1_s1120
                                 - 0.232999966201995 z_k1_s1121
                                 - 0.251999411617916 z_k1_s1122
                                 - 0.131999968066373 z_k1_s1123
                                 - 0.108999929679075 z_k1_s1124
                                 - 0.155999919496647 z_k1_s1125
                                 - 0.131999968066373 z_k1_s1126
                                 - 0.275999206248277 z_k1_s1127
                                 - 0.388000981165372 z_k1_s1130
                                 - 0.305000587276438 z_k1_s1131
                                 - 0.319000826679828 z_k1_s1132
                                 - 0.228000628561227 z_k1_s1133
                                 - 0.694001834971255 z_k1_s1134
                                 - 0.656001018875918 z_k1_s1135
                                 - 0.462000528293232 z_k1_s1136
                                 - 0.828001853826528 z_k1_s1137
                                 - 0.950996874530796 z_k1_s1140
                                 - 0.516999004221478 z_k1_s1141
                                 - 0.91299753786493 z_k1_s1142
                                 - 0.960998103650462 z_k1_s1143
                                 - 0.310999287502905 z_k1_s1144
                                 - 0.459998932424905 z_k1_s1145
                                 - 0.489998885991698 z_k1_s1146
                                 - 0.730996503842465 z_k1_s1147
                                 - 0.155000301776571 z_k1_s1150
                                 - 0.0969999749633975 z_k1_s1151
                                 - 0.10100007726086 z_k1_s1152
                                 - 0.180000748385706 z_k1_s1153
                                 - 0.144999968932914 z_k1_s1154
                                 - 0.462000173256602 z_k1_s1155
                                 - 0.273000312651633 z_k1_s1156
                                 - 0.129000022343076 z_k1_s1157
                                 - 0.142000626853198 z_k1_s1160
                                 - 0.215000966259498 z_k1_s1161
                                 - 0.277001439528702 z_k1_s1162
                                 - 0.493002474128302 z_k1_s1163
                                 - 0.168000853888801 z_k1_s1164
                                 - 0.349001046391885 z_k1_s1165
                                 - 0.265001505051505 z_k1_s1166
                                 - 0.305002024645512 z_k1_s1167
                                 - 0.360999809799516 z_k1_s1170
                                 - 0.114999932264668 z_k1_s1171
                                 - 0.297999418252924 z_k1_s1172
                                 - 0.164999865431927 z_k1_s1173
                                 - 0.15799966270458 z_k1_s1174
                                 - 0.174999336185546 z_k1_s1175
                                 - 0.0829999062536445 z_k1_s1176
                                 - 0.0780000275768814 z_k1_s1177 >= 0
 z_k1_s0000_constraint:          var_1 + z_k1_s0000 + 1.01536 w_k1_ADM_s000
                                 + 0.97629 w_k1_AXP_s000
                                 + 1.16658 w_k1_DIS_s000
                                 + 1.10109 w_k1_FDX_s000
                                 + 0.789591 w_k1_IBM_s000
                                 + 1.44814 w_k1_INTC_s000
                                 + 0.973903 w_k1_KO_s000
                                 + 0.486831 w_k1_MAT_s000
                                 + 1.22615 w_k1_MSFT_s000
                                 + 1.08293 w_k1_PG_s000 >= 1
 z_k1_s0001_constraint:          var_1 + z_k1_s0001 + 0.889781 w_k1_ADM_s000
                                 + 0.768841 w_k1_AXP_s000
                                 + 0.593785 w_k1_DIS_s000
                                 + 0.691944 w_k1_FDX_s000
                                 + 1.05791 w_k1_IBM_s000
                                 + 0.785885 w_k1_INTC_s000
                                 + 0.876232 w_k1_KO_s000
                                 + 0.944478 w_k1_MAT_s000
                                 + 0.786557 w_k1_MSFT_s000
                                 + 0.86525 w_k1_PG_s000 >= 1
 z_k1_s0002_constraint:          var_1 + z_k1_s0002 + 0.987802 w_k1_ADM_s000
                                 + 1.29718 w_k1_AXP_s000
                                 + 1.09467 w_k1_DIS_s000
                                 + 1.02409 w_k1_FDX_s000
                                 + 1.67125 w_k1_IBM_s000
                                 + 1.27679 w_k1_INTC_s000
                                 + 1.27966 w_k1_KO_s000 + 1.30494 w_k1_MAT_s000
                                 + 1.21601 w_k1_MSFT_s000
                                 + 1.20893 w_k1_PG_s000 >= 1
 z_k1_s0003_constraint:          var_1 + z_k1_s0003 + 0.844215 w_k1_ADM_s000
                                 + 1.06999 w_k1_AXP_s000
                                 + 1.08443 w_k1_DIS_s000
                                 + 1.10372 w_k1_FDX_s000
                                 + 1.20607 w_k1_IBM_s000
                                 + 1.17394 w_k1_INTC_s000
                                 + 1.05984 w_k1_KO_s000 + 1.07082 w_k1_MAT_s000
                                 + 1.09433 w_k1_MSFT_s000
                                 + 1.07476 w_k1_PG_s000 >= 1
 z_k1_s0004_constraint:          var_1 + z_k1_s0004 + 0.831638 w_k1_ADM_s000
                                 + 0.715707 w_k1_AXP_s000
                                 + 0.646933 w_k1_DIS_s000
                                 + 0.843975 w_k1_FDX_s000
                                 + 0.619763 w_k1_IBM_s000
                                 + 0.869212 w_k1_INTC_s000
                                 + 0.69491 w_k1_KO_s000
                                 + 0.543768 w_k1_MAT_s000
                                 + 0.904014 w_k1_MSFT_s000
                                 + 0.847739 w_k1_PG_s000 >= 1
 z_k1_s0005_constraint:          var_1 + z_k1_s0005 + 1.43071 w_k1_ADM_s000
                                 + 1.55961 w_k1_AXP_s000
                                 + 1.51967 w_k1_DIS_s000
                                 + 1.37537 w_k1_FDX_s000
                                 + 1.16637 w_k1_IBM_s000
                                 + 1.41492 w_k1_INTC_s000
                                 + 1.24354 w_k1_KO_s000 + 1.01826 w_k1_MAT_s000
                                 + 1.29343 w_k1_MSFT_s000
                                 + 1.09256 w_k1_PG_s000 >= 1
 z_k1_s0006_constraint:          var_1 + z_k1_s0006 + 1.02863 w_k1_ADM_s000
                                 + 0.973004 w_k1_AXP_s000
                                 + 0.847218 w_k1_DIS_s000
                                 + 0.966552 w_k1_FDX_s000
                                 + 0.816062 w_k1_IBM_s000
                                 + 1.09955 w_k1_INTC_s000
                                 + 0.90139 w_k1_KO_s000
                                 + 0.826198 w_k1_MAT_s000
                                 + 1.01989 w_k1_MSFT_s000
                                 + 0.887572 w_k1_PG_s000 >= 1
 z_k1_s0007_constraint:          var_1 + z_k1_s0007 + 0.877402 w_k1_ADM_s000
                                 + 0.976119 w_k1_AXP_s000
                                 + 0.775707 w_k1_DIS_s000
                                 + 0.862513 w_k1_FDX_s000
                                 + 1.06906 w_k1_IBM_s000
                                 + 1.02369 w_k1_INTC_s000
                                 + 0.927819 w_k1_KO_s000
                                 + 0.882396 w_k1_MAT_s000
                                 + 0.933167 w_k1_MSFT_s000
                                 + 0.954408 w_k1_PG_s000 >= 1
 z_k1_s0010_constraint:          var_1 + z_k1_s0010 + 1.10437 w_k1_ADM_s001
                                 + 1.09311 w_k1_AXP_s001
                                 + 0.981039 w_k1_DIS_s001
                                 + 0.978597 w_k1_FDX_s001
                                 + 1.11368 w_k1_IBM_s001
                                 + 0.841216 w_k1_INTC_s001 + 1.061 w_k1_KO_s001
                                 + 1.3771 w_k1_MAT_s001
                                 + 1.07945 w_k1_MSFT_s001
                                 + 1.13563 w_k1_PG_s001 >= 1
 z_k1_s0011_constraint:          var_1 + z_k1_s0011 + 1.29202 w_k1_ADM_s001
                                 + 0.92525 w_k1_AXP_s001
                                 + 1.18132 w_k1_DIS_s001
                                 + 0.979794 w_k1_FDX_s001
                                 + 1.48556 w_k1_IBM_s001
                                 + 1.02863 w_k1_INTC_s001
                                 + 1.31556 w_k1_KO_s001 + 1.45884 w_k1_MAT_s001
                                 + 1.14026 w_k1_MSFT_s001 + 1.0238 w_k1_PG_s001
                                 >= 1
 z_k1_s0012_constraint:          var_1 + z_k1_s0012 + 1.25218 w_k1_ADM_s001
                                 + 1.41283 w_k1_AXP_s001
                                 + 1.48352 w_k1_DIS_s001
                                 + 1.28315 w_k1_FDX_s001
                                 + 1.32578 w_k1_IBM_s001
                                 + 1.28939 w_k1_INTC_s001
                                 + 1.31092 w_k1_KO_s001 + 1.28422 w_k1_MAT_s001
                                 + 1.33529 w_k1_MSFT_s001
                                 + 1.28368 w_k1_PG_s001 >= 1
 z_k1_s0013_constraint:          var_1 + z_k1_s0013 + 1.44831 w_k1_ADM_s001
                                 + 1.07725 w_k1_AXP_s001
                                 + 1.45537 w_k1_DIS_s001
                                 + 1.39364 w_k1_FDX_s001
                                 + 0.876533 w_k1_IBM_s001
                                 + 1.20584 w_k1_INTC_s001
                                 + 1.28518 w_k1_KO_s001 + 1.14407 w_k1_MAT_s001
                                 + 1.5162 w_k1_MSFT_s001 + 1.29061 w_k1_PG_s001
                                 >= 1
 z_k1_s0014_constraint:          var_1 + z_k1_s0014 + 1.17496 w_k1_ADM_s001
                                 + 1.10227 w_k1_AXP_s001
                                 + 1.02957 w_k1_DIS_s001
                                 + 1.13774 w_k1_FDX_s001
                                 + 0.784237 w_k1_IBM_s001
                                 + 1.03461 w_k1_INTC_s001
                                 + 0.981477 w_k1_KO_s001
                                 + 0.755292 w_k1_MAT_s001
                                 + 1.0259 w_k1_MSFT_s001
                                 + 0.953874 w_k1_PG_s001 >= 1
 z_k1_s0015_constraint:          var_1 + z_k1_s0015 + 0.885831 w_k1_ADM_s001
                                 + 0.760876 w_k1_AXP_s001
                                 + 0.815943 w_k1_DIS_s001
                                 + 0.897074 w_k1_FDX_s001
                                 + 1.0264 w_k1_IBM_s001
                                 + 0.844905 w_k1_INTC_s001
                                 + 0.966592 w_k1_KO_s001
                                 + 0.897915 w_k1_MAT_s001
                                 + 0.921179 w_k1_MSFT_s001
                                 + 0.992817 w_k1_PG_s001 >= 1
 z_k1_s0016_constraint:          var_1 + z_k1_s0016 + 1.03524 w_k1_ADM_s001
                                 + 0.974125 w_k1_AXP_s001
                                 + 0.618629 w_k1_DIS_s001
                                 + 0.883208 w_k1_FDX_s001
                                 + 0.85487 w_k1_IBM_s001
                                 + 0.866896 w_k1_INTC_s001
                                 + 0.894397 w_k1_KO_s001
                                 + 1.59671 w_k1_MAT_s001
                                 + 0.858435 w_k1_MSFT_s001
                                 + 1.00639 w_k1_PG_s001 >= 1
 z_k1_s0017_constraint:          var_1 + z_k1_s0017 + 0.971363 w_k1_ADM_s001
                                 + 0.722863 w_k1_AXP_s001
                                 + 0.566634 w_k1_DIS_s001
                                 + 0.757173 w_k1_FDX_s001
                                 + 0.719219 w_k1_IBM_s001
                                 + 0.858804 w_k1_INTC_s001
                                 + 0.709399 w_k1_KO_s001
                                 + 0.656171 w_k1_MAT_s001
                                 + 0.926111 w_k1_MSFT_s001
                                 + 0.851889 w_k1_PG_s001 >= 1
 z_k1_s0020_constraint:          var_1 + z_k1_s0020 + 1.04416 w_k1_ADM_s002
                                 + 0.915091 w_k1_AXP_s002
                                 + 1.18423 w_k1_DIS_s002
                                 + 1.06084 w_k1_FDX_s002
                                 + 1.29147 w_k1_IBM_s002
                                 + 1.07798 w_k1_INTC_s002
                                 + 1.19187 w_k1_KO_s002 + 1.05403 w_k1_MAT_s002
                                 + 1.07153 w_k1_MSFT_s002
                                 + 1.12363 w_k1_PG_s002 >= 1
 z_k1_s0021_constraint:          var_1 + z_k1_s0021 + 1.07386 w_k1_ADM_s002
                                 + 0.806292 w_k1_AXP_s002
                                 + 0.847747 w_k1_DIS_s002
                                 + 0.978989 w_k1_FDX_s002
                                 + 0.937111 w_k1_IBM_s002
                                 + 0.8933 w_k1_INTC_s002
                                 + 0.832316 w_k1_KO_s002
                                 + 0.670637 w_k1_MAT_s002
                                 + 0.929038 w_k1_MSFT_s002
                                 + 1.00592 w_k1_PG_s002 >= 1
 z_k1_s0022_constraint:          var_1 + z_k1_s0022 + 0.799521 w_k1_ADM_s002
                                 + 0.937082 w_k1_AXP_s002
                                 + 0.929713 w_k1_DIS_s002
                                 + 0.897503 w_k1_FDX_s002
                                 + 1.21967 w_k1_IBM_s002
                                 + 0.871785 w_k1_INTC_s002
                                 + 1.0259 w_k1_KO_s002 + 0.938834 w_k1_MAT_s002
                                 + 0.810836 w_k1_MSFT_s002
                                 + 0.973197 w_k1_PG_s002 >= 1
 z_k1_s0023_constraint:          var_1 + z_k1_s0023 + 0.564944 w_k1_ADM_s002
                                 + 0.649656 w_k1_AXP_s002
                                 + 0.649299 w_k1_DIS_s002
                                 + 0.754262 w_k1_FDX_s002
                                 + 1.03528 w_k1_IBM_s002
                                 + 0.62543 w_k1_INTC_s002
                                 + 0.745998 w_k1_KO_s002
                                 + 0.844283 w_k1_MAT_s002
                                 + 0.676273 w_k1_MSFT_s002
                                 + 0.797296 w_k1_PG_s002 >= 1
 z_k1_s0024_constraint:          var_1 + z_k1_s0024 + 1.13713 w_k1_ADM_s002
                                 + 1.39987 w_k1_AXP_s002
                                 + 1.37934 w_k1_DIS_s002
                                 + 1.22888 w_k1_FDX_s002 + 1.3388 w_k1_IBM_s002
                                 + 1.38672 w_k1_INTC_s002
                                 + 1.13729 w_k1_KO_s002 + 1.16015 w_k1_MAT_s002
                                 + 1.21405 w_k1_MSFT_s002
                                 + 1.14022 w_k1_PG_s002 >= 1
 z_k1_s0025_constraint:          var_1 + z_k1_s0025 + 0.981777 w_k1_ADM_s002
                                 + 1.08027 w_k1_AXP_s002
                                 + 1.21375 w_k1_DIS_s002
                                 + 1.22777 w_k1_FDX_s002
                                 + 0.820111 w_k1_IBM_s002
                                 + 1.18223 w_k1_INTC_s002
                                 + 0.992038 w_k1_KO_s002
                                 + 0.796059 w_k1_MAT_s002
                                 + 1.1173 w_k1_MSFT_s002
                                 + 0.974382 w_k1_PG_s002 >= 1
 z_k1_s0026_constraint:          var_1 + z_k1_s0026 + 0.772422 w_k1_ADM_s002
                                 + 0.754889 w_k1_AXP_s002
                                 + 0.659301 w_k1_DIS_s002
                                 + 0.852206 w_k1_FDX_s002
                                 + 0.763944 w_k1_IBM_s002
                                 + 0.944181 w_k1_INTC_s002
                                 + 0.697859 w_k1_KO_s002
                                 + 0.588777 w_k1_MAT_s002
                                 + 0.824674 w_k1_MSFT_s002
                                 + 0.848785 w_k1_PG_s002 >= 1
 z_k1_s0027_constraint:          var_1 + z_k1_s0027 + 0.954269 w_k1_ADM_s002
                                 + 0.772031 w_k1_AXP_s002
                                 + 0.957004 w_k1_DIS_s002
                                 + 0.981566 w_k1_FDX_s002
                                 + 0.69605 w_k1_IBM_s002
                                 + 0.964196 w_k1_INTC_s002
                                 + 0.858678 w_k1_KO_s002
                                 + 0.537738 w_k1_MAT_s002
                                 + 0.9735 w_k1_MSFT_s002
                                 + 0.950689 w_k1_PG_s002 >= 1
 z_k1_s0030_constraint:          var_1 + z_k1_s0030 + 1.53649 w_k1_ADM_s003
                                 + 1.37071 w_k1_AXP_s003
                                 + 1.17468 w_k1_DIS_s003
                                 + 1.32408 w_k1_FDX_s003
                                 + 0.805567 w_k1_IBM_s003
                                 + 1.18663 w_k1_INTC_s003
                                 + 1.04087 w_k1_KO_s003 + 1.09231 w_k1_MAT_s003
                                 + 1.26925 w_k1_MSFT_s003
                                 + 1.05839 w_k1_PG_s003 >= 1
 z_k1_s0031_constraint:          var_1 + z_k1_s0031 + 0.908908 w_k1_ADM_s003
                                 + 0.960631 w_k1_AXP_s003
                                 + 1.02485 w_k1_DIS_s003
                                 + 0.958745 w_k1_FDX_s003
                                 + 1.08074 w_k1_IBM_s003
                                 + 1.02162 w_k1_INTC_s003
                                 + 1.04379 w_k1_KO_s003
                                 + 0.867384 w_k1_MAT_s003
                                 + 1.05549 w_k1_MSFT_s003
                                 + 0.902899 w_k1_PG_s003 >= 1
 z_k1_s0032_constraint:          var_1 + z_k1_s0032 + 0.992873 w_k1_ADM_s003
                                 + 0.734724 w_k1_AXP_s003
                                 + 1.0353 w_k1_DIS_s003
                                 + 0.987617 w_k1_FDX_s003
                                 + 0.83915 w_k1_IBM_s003
                                 + 1.12179 w_k1_INTC_s003
                                 + 0.962266 w_k1_KO_s003
                                 + 0.711501 w_k1_MAT_s003
                                 + 1.13156 w_k1_MSFT_s003
                                 + 0.849807 w_k1_PG_s003 >= 1
 z_k1_s0033_constraint:          var_1 + z_k1_s0033 + 1.18155 w_k1_ADM_s003
                                 + 1.31067 w_k1_AXP_s003
                                 + 1.23335 w_k1_DIS_s003
                                 + 1.17611 w_k1_FDX_s003
                                 + 1.06499 w_k1_IBM_s003
                                 + 1.37843 w_k1_INTC_s003
                                 + 1.28877 w_k1_KO_s003
                                 + 0.870916 w_k1_MAT_s003
                                 + 1.20221 w_k1_MSFT_s003
                                 + 1.17743 w_k1_PG_s003 >= 1
 z_k1_s0034_constraint:          var_1 + z_k1_s0034 + 1.12745 w_k1_ADM_s003
                                 + 1.37586 w_k1_AXP_s003
                                 + 1.37696 w_k1_DIS_s003
                                 + 1.15816 w_k1_FDX_s003
                                 + 1.45995 w_k1_IBM_s003
                                 + 1.4869 w_k1_INTC_s003 + 1.2787 w_k1_KO_s003
                                 + 1.33231 w_k1_MAT_s003
                                 + 1.19551 w_k1_MSFT_s003
                                 + 1.10575 w_k1_PG_s003 >= 1
 z_k1_s0035_constraint:          var_1 + z_k1_s0035 + 0.862096 w_k1_ADM_s003
                                 + 0.890899 w_k1_AXP_s003
                                 + 0.85235 w_k1_DIS_s003
                                 + 0.855476 w_k1_FDX_s003
                                 + 1.23204 w_k1_IBM_s003
                                 + 0.985901 w_k1_INTC_s003
                                 + 1.1234 w_k1_KO_s003 + 1.11052 w_k1_MAT_s003
                                 + 0.890182 w_k1_MSFT_s003
                                 + 0.916133 w_k1_PG_s003 >= 1
 z_k1_s0036_constraint:          var_1 + z_k1_s0036 + 0.742968 w_k1_ADM_s003
                                 + 0.822003 w_k1_AXP_s003
                                 + 0.6985 w_k1_DIS_s003
                                 + 0.825672 w_k1_FDX_s003
                                 + 0.698994 w_k1_IBM_s003
                                 + 1.05505 w_k1_INTC_s003
                                 + 0.739865 w_k1_KO_s003
                                 + 0.569839 w_k1_MAT_s003
                                 + 0.840236 w_k1_MSFT_s003
                                 + 0.793364 w_k1_PG_s003 >= 1
 z_k1_s0037_constraint:          var_1 + z_k1_s0037 + 1.09531 w_k1_ADM_s003
                                 + 0.939895 w_k1_AXP_s003
                                 + 0.890598 w_k1_DIS_s003
                                 + 1.06833 w_k1_FDX_s003
                                 + 0.810763 w_k1_IBM_s003
                                 + 1.1903 w_k1_INTC_s003
                                 + 0.890557 w_k1_KO_s003
                                 + 0.610503 w_k1_MAT_s003
                                 + 0.992571 w_k1_MSFT_s003
                                 + 0.969761 w_k1_PG_s003 >= 1
 z_k1_s0040_constraint:          var_1 + z_k1_s0040 + 0.63634 w_k1_ADM_s004
                                 + 0.778219 w_k1_AXP_s004
                                 + 0.689392 w_k1_DIS_s004
                                 + 0.803948 w_k1_FDX_s004
                                 + 1.11066 w_k1_IBM_s004
                                 + 1.06997 w_k1_INTC_s004
                                 + 0.886479 w_k1_KO_s004
                                 + 0.833366 w_k1_MAT_s004
                                 + 0.815962 w_k1_MSFT_s004
                                 + 1.00856 w_k1_PG_s004 >= 1
 z_k1_s0041_constraint:          var_1 + z_k1_s0041 + 0.947755 w_k1_ADM_s004
                                 + 0.898866 w_k1_AXP_s004
                                 + 0.927229 w_k1_DIS_s004
                                 + 0.904227 w_k1_FDX_s004
                                 + 1.1025 w_k1_IBM_s004
                                 + 1.06541 w_k1_INTC_s004
                                 + 0.871418 w_k1_KO_s004
                                 + 0.783974 w_k1_MAT_s004
                                 + 0.898449 w_k1_MSFT_s004
                                 + 0.944748 w_k1_PG_s004 >= 1
 z_k1_s0042_constraint:          var_1 + z_k1_s0042 + 0.97457 w_k1_ADM_s004
                                 + 0.863683 w_k1_AXP_s004
                                 + 1.09907 w_k1_DIS_s004
                                 + 0.996292 w_k1_FDX_s004
                                 + 1.45706 w_k1_IBM_s004
                                 + 1.11185 w_k1_INTC_s004
                                 + 1.08573 w_k1_KO_s004 + 1.21932 w_k1_MAT_s004
                                 + 1.02875 w_k1_MSFT_s004
                                 + 1.06385 w_k1_PG_s004 >= 1
 z_k1_s0043_constraint:          var_1 + z_k1_s0043 + 1.20856 w_k1_ADM_s004
                                 + 1.20569 w_k1_AXP_s004
                                 + 1.17681 w_k1_DIS_s004
                                 + 1.21475 w_k1_FDX_s004
                                 + 0.853135 w_k1_IBM_s004
                                 + 1.25712 w_k1_INTC_s004
                                 + 0.806889 w_k1_KO_s004
                                 + 0.600022 w_k1_MAT_s004
                                 + 1.0526 w_k1_MSFT_s004
                                 + 0.999977 w_k1_PG_s004 >= 1
 z_k1_s0044_constraint:          var_1 + z_k1_s0044 + 0.864607 w_k1_ADM_s004
                                 + 0.886575 w_k1_AXP_s004
                                 + 0.683033 w_k1_DIS_s004
                                 + 0.917891 w_k1_FDX_s004
                                 + 0.859656 w_k1_IBM_s004
                                 + 1.24178 w_k1_INTC_s004
                                 + 0.752467 w_k1_KO_s004
                                 + 0.613914 w_k1_MAT_s004
                                 + 0.920385 w_k1_MSFT_s004
                                 + 0.904825 w_k1_PG_s004 >= 1
 z_k1_s0045_constraint:          var_1 + z_k1_s0045 + 1.09107 w_k1_ADM_s004
                                 + 1.3078 w_k1_AXP_s004 + 1.62869 w_k1_DIS_s004
                                 + 1.27618 w_k1_FDX_s004
                                 + 1.43043 w_k1_IBM_s004
                                 + 1.57546 w_k1_INTC_s004
                                 + 1.30143 w_k1_KO_s004 + 1.04195 w_k1_MAT_s004
                                 + 1.31287 w_k1_MSFT_s004
                                 + 1.39646 w_k1_PG_s004 >= 1
 z_k1_s0046_constraint:          var_1 + z_k1_s0046 + 0.80724 w_k1_ADM_s004
                                 + 0.723571 w_k1_AXP_s004
                                 + 0.784934 w_k1_DIS_s004
                                 + 0.820766 w_k1_FDX_s004
                                 + 0.749749 w_k1_IBM_s004
                                 + 0.954148 w_k1_INTC_s004
                                 + 0.763306 w_k1_KO_s004
                                 + 0.539638 w_k1_MAT_s004
                                 + 0.784084 w_k1_MSFT_s004
                                 + 0.871109 w_k1_PG_s004 >= 1
 z_k1_s0047_constraint:          var_1 + z_k1_s0047 + 0.676076 w_k1_ADM_s004
                                 + 0.574572 w_k1_AXP_s004
                                 + 0.49633 w_k1_DIS_s004
                                 + 0.649232 w_k1_FDX_s004
                                 + 0.734067 w_k1_IBM_s004
                                 + 1.00224 w_k1_INTC_s004
                                 + 0.588483 w_k1_KO_s004
                                 + 0.419626 w_k1_MAT_s004
                                 + 0.752941 w_k1_MSFT_s004
                                 + 0.775205 w_k1_PG_s004 >= 1
 z_k1_s0050_constraint:          var_1 + z_k1_s0050 + 0.686417 w_k1_ADM_s005
                                 + 1.21415 w_k1_AXP_s005
                                 + 0.902711 w_k1_DIS_s005
                                 + 1.11598 w_k1_FDX_s005
                                 + 1.00401 w_k1_IBM_s005
                                 + 1.2886 w_k1_INTC_s005
                                 + 0.889299 w_k1_KO_s005
                                 + 1.17878 w_k1_MAT_s005
                                 + 1.15533 w_k1_MSFT_s005
                                 + 1.11729 w_k1_PG_s005 >= 1
 z_k1_s0051_constraint:          var_1 + z_k1_s0051 + 0.676222 w_k1_ADM_s005
                                 + 0.872372 w_k1_AXP_s005
                                 + 0.49206 w_k1_DIS_s005
                                 + 0.825284 w_k1_FDX_s005
                                 + 0.638121 w_k1_IBM_s005
                                 + 0.795541 w_k1_INTC_s005
                                 + 0.535221 w_k1_KO_s005
                                 + 0.751494 w_k1_MAT_s005
                                 + 0.754305 w_k1_MSFT_s005
                                 + 0.804156 w_k1_PG_s005 >= 1
 z_k1_s0052_constraint:          var_1 + z_k1_s0052 + 0.874601 w_k1_ADM_s005
                                 + 0.74158 w_k1_AXP_s005
                                 + 0.86585 w_k1_DIS_s005
                                 + 0.934902 w_k1_FDX_s005
                                 + 0.753489 w_k1_IBM_s005
                                 + 0.965459 w_k1_INTC_s005
                                 + 0.795565 w_k1_KO_s005
                                 + 0.57421 w_k1_MAT_s005
                                 + 1.05314 w_k1_MSFT_s005
                                 + 1.01937 w_k1_PG_s005 >= 1
 z_k1_s0053_constraint:          var_1 + z_k1_s0053 + 0.923194 w_k1_ADM_s005
                                 + 1.09628 w_k1_AXP_s005
                                 + 1.22864 w_k1_DIS_s005
                                 + 1.00809 w_k1_FDX_s005
                                 + 1.56598 w_k1_IBM_s005
                                 + 1.14098 w_k1_INTC_s005
                                 + 1.28054 w_k1_KO_s005 + 1.61442 w_k1_MAT_s005
                                 + 1.22802 w_k1_MSFT_s005
                                 + 1.27169 w_k1_PG_s005 >= 1
 z_k1_s0054_constraint:          var_1 + z_k1_s0054 + 0.901644 w_k1_ADM_s005
                                 + 0.943864 w_k1_AXP_s005
                                 + 0.943438 w_k1_DIS_s005
                                 + 1.21112 w_k1_FDX_s005
                                 + 0.648378 w_k1_IBM_s005
                                 + 0.987271 w_k1_INTC_s005
                                 + 0.891837 w_k1_KO_s005
                                 + 0.709899 w_k1_MAT_s005
                                 + 1.21746 w_k1_MSFT_s005
                                 + 1.19717 w_k1_PG_s005 >= 1
 z_k1_s0055_constraint:          var_1 + z_k1_s0055 + 0.663727 w_k1_ADM_s005
                                 + 0.818307 w_k1_AXP_s005
                                 + 0.74125 w_k1_DIS_s005
                                 + 0.774153 w_k1_FDX_s005
                                 + 1.27479 w_k1_IBM_s005
                                 + 0.889003 w_k1_INTC_s005
                                 + 1.03257 w_k1_KO_s005
                                 + 0.982335 w_k1_MAT_s005
                                 + 0.793834 w_k1_MSFT_s005
                                 + 0.951072 w_k1_PG_s005 >= 1
 z_k1_s0056_constraint:          var_1 + z_k1_s0056 + 1.15932 w_k1_ADM_s005
                                 + 1.24377 w_k1_AXP_s005
                                 + 1.41281 w_k1_DIS_s005
                                 + 1.41643 w_k1_FDX_s005
                                 + 1.01152 w_k1_IBM_s005
                                 + 1.44152 w_k1_INTC_s005
                                 + 1.10383 w_k1_KO_s005 + 1.10479 w_k1_MAT_s005
                                 + 1.37517 w_k1_MSFT_s005
                                 + 1.31516 w_k1_PG_s005 >= 1
 z_k1_s0057_constraint:          var_1 + z_k1_s0057 + 0.78386 w_k1_ADM_s005
                                 + 0.749517 w_k1_AXP_s005
                                 + 0.661701 w_k1_DIS_s005
                                 + 0.85256 w_k1_FDX_s005
                                 + 0.854813 w_k1_IBM_s005
                                 + 0.985936 w_k1_INTC_s005
                                 + 0.826347 w_k1_KO_s005
                                 + 0.679301 w_k1_MAT_s005
                                 + 0.823253 w_k1_MSFT_s005
                                 + 0.906481 w_k1_PG_s005 >= 1
 z_k1_s0060_constraint:          var_1 + z_k1_s0060 + 0.968826 w_k1_ADM_s006
                                 + 0.856401 w_k1_AXP_s006
                                 + 0.672083 w_k1_DIS_s006
                                 + 0.943205 w_k1_FDX_s006
                                 + 0.609823 w_k1_IBM_s006
                                 + 0.989222 w_k1_INTC_s006
                                 + 0.666235 w_k1_KO_s006
                                 + 0.599757 w_k1_MAT_s006
                                 + 0.946938 w_k1_MSFT_s006
                                 + 0.894986 w_k1_PG_s006 >= 1
 z_k1_s0061_constraint:          var_1 + z_k1_s0061 + 0.68575 w_k1_ADM_s006
                                 + 0.531425 w_k1_AXP_s006
                                 + 0.512822 w_k1_DIS_s006
                                 + 0.694342 w_k1_FDX_s006
                                 + 0.849885 w_k1_IBM_s006
                                 + 0.727143 w_k1_INTC_s006
                                 + 0.699848 w_k1_KO_s006
                                 + 0.578657 w_k1_MAT_s006
                                 + 0.748558 w_k1_MSFT_s006
                                 + 0.82752 w_k1_PG_s006 >= 1
 z_k1_s0062_constraint:          var_1 + z_k1_s0062 + 0.99172 w_k1_ADM_s006
                                 + 0.916633 w_k1_AXP_s006
                                 + 0.909747 w_k1_DIS_s006
                                 + 0.931968 w_k1_FDX_s006
                                 + 1.25895 w_k1_IBM_s006
                                 + 0.90661 w_k1_INTC_s006
                                 + 1.11234 w_k1_KO_s006 + 1.11338 w_k1_MAT_s006
                                 + 0.998595 w_k1_MSFT_s006
                                 + 1.11678 w_k1_PG_s006 >= 1
 z_k1_s0063_constraint:          var_1 + z_k1_s0063 + 0.95302 w_k1_ADM_s006
                                 + 0.978317 w_k1_AXP_s006
                                 + 0.89594 w_k1_DIS_s006
                                 + 1.04747 w_k1_FDX_s006
                                 + 0.876262 w_k1_IBM_s006
                                 + 0.978825 w_k1_INTC_s006
                                 + 0.888171 w_k1_KO_s006
                                 + 0.862497 w_k1_MAT_s006
                                 + 1.06241 w_k1_MSFT_s006
                                 + 1.10444 w_k1_PG_s006 >= 1
 z_k1_s0064_constraint:          var_1 + z_k1_s0064 + 1.43977 w_k1_ADM_s006
                                 + 1.28784 w_k1_AXP_s006 + 1.4297 w_k1_DIS_s006
                                 + 1.37349 w_k1_FDX_s006
                                 + 1.08588 w_k1_IBM_s006
                                 + 1.22677 w_k1_INTC_s006
                                 + 1.23261 w_k1_KO_s006
                                 + 0.907756 w_k1_MAT_s006
                                 + 1.25391 w_k1_MSFT_s006
                                 + 1.36965 w_k1_PG_s006 >= 1
 z_k1_s0065_constraint:          var_1 + z_k1_s0065 + 1.03587 w_k1_ADM_s006
                                 + 1.09907 w_k1_AXP_s006
                                 + 1.23469 w_k1_DIS_s006
                                 + 1.04521 w_k1_FDX_s006
                                 + 1.65953 w_k1_IBM_s006
                                 + 1.09669 w_k1_INTC_s006
                                 + 1.43098 w_k1_KO_s006 + 1.49818 w_k1_MAT_s006
                                 + 1.169 w_k1_MSFT_s006 + 1.30834 w_k1_PG_s006
                                 >= 1
 z_k1_s0066_constraint:          var_1 + z_k1_s0066 + 0.869624 w_k1_ADM_s006
                                 + 0.993376 w_k1_AXP_s006
                                 + 0.715223 w_k1_DIS_s006
                                 + 0.909747 w_k1_FDX_s006
                                 + 1.01191 w_k1_IBM_s006
                                 + 0.97691 w_k1_INTC_s006
                                 + 0.838819 w_k1_KO_s006
                                 + 0.643562 w_k1_MAT_s006
                                 + 0.824538 w_k1_MSFT_s006
                                 + 0.940072 w_k1_PG_s006 >= 1
 z_k1_s0067_constraint:          var_1 + z_k1_s0067 + 0.697156 w_k1_ADM_s006
                                 + 0.669522 w_k1_AXP_s006
                                 + 0.645581 w_k1_DIS_s006
                                 + 0.668361 w_k1_FDX_s006
                                 + 1.25148 w_k1_IBM_s006
                                 + 0.789094 w_k1_INTC_s006
                                 + 0.762039 w_k1_KO_s006
                                 + 0.969166 w_k1_MAT_s006
                                 + 0.679734 w_k1_MSFT_s006
                                 + 0.85417 w_k1_PG_s006 >= 1
 z_k1_s0070_constraint:          var_1 + z_k1_s0070 + 1.05762 w_k1_ADM_s007
                                 + 1.25033 w_k1_AXP_s007
                                 + 1.11394 w_k1_DIS_s007
                                 + 1.06632 w_k1_FDX_s007 + 1.1908 w_k1_IBM_s007
                                 + 1.1247 w_k1_INTC_s007 + 1.05562 w_k1_KO_s007
                                 + 0.863742 w_k1_MAT_s007
                                 + 1.01359 w_k1_MSFT_s007
                                 + 1.10311 w_k1_PG_s007 >= 1
 z_k1_s0071_constraint:          var_1 + z_k1_s0071 + 0.711475 w_k1_ADM_s007
                                 + 0.762362 w_k1_AXP_s007
                                 + 0.729762 w_k1_DIS_s007
                                 + 0.791106 w_k1_FDX_s007
                                 + 0.973483 w_k1_IBM_s007
                                 + 0.960442 w_k1_INTC_s007
                                 + 0.860216 w_k1_KO_s007
                                 + 0.915272 w_k1_MAT_s007
                                 + 0.819987 w_k1_MSFT_s007
                                 + 0.967037 w_k1_PG_s007 >= 1
 z_k1_s0072_constraint:          var_1 + z_k1_s0072 + 0.889006 w_k1_ADM_s007
                                 + 0.798855 w_k1_AXP_s007
                                 + 0.952757 w_k1_DIS_s007
                                 + 0.841088 w_k1_FDX_s007
                                 + 1.37794 w_k1_IBM_s007
                                 + 1.02421 w_k1_INTC_s007
                                 + 1.19228 w_k1_KO_s007
                                 + 0.923728 w_k1_MAT_s007
                                 + 0.897671 w_k1_MSFT_s007
                                 + 1.08029 w_k1_PG_s007 >= 1
 z_k1_s0073_constraint:          var_1 + z_k1_s0073 + 1.16877 w_k1_ADM_s007
                                 + 1.46856 w_k1_AXP_s007
                                 + 1.78982 w_k1_DIS_s007
                                 + 1.38505 w_k1_FDX_s007
                                 + 1.43645 w_k1_IBM_s007
                                 + 1.26179 w_k1_INTC_s007
                                 + 1.66906 w_k1_KO_s007 + 1.30333 w_k1_MAT_s007
                                 + 1.31449 w_k1_MSFT_s007
                                 + 1.34527 w_k1_PG_s007 >= 1
 z_k1_s0074_constraint:          var_1 + z_k1_s0074 + 1.04735 w_k1_ADM_s007
                                 + 0.800774 w_k1_AXP_s007
                                 + 1.34178 w_k1_DIS_s007
                                 + 1.08838 w_k1_FDX_s007
                                 + 0.844727 w_k1_IBM_s007
                                 + 1.18124 w_k1_INTC_s007
                                 + 1.16322 w_k1_KO_s007
                                 + 0.676537 w_k1_MAT_s007
                                 + 1.13376 w_k1_MSFT_s007
                                 + 1.13294 w_k1_PG_s007 >= 1
 z_k1_s0075_constraint:          var_1 + z_k1_s0075 + 1.02814 w_k1_ADM_s007
                                 + 1.23978 w_k1_AXP_s007
                                 + 1.15258 w_k1_DIS_s007
                                 + 1.01478 w_k1_FDX_s007
                                 + 1.56486 w_k1_IBM_s007
                                 + 1.06176 w_k1_INTC_s007
                                 + 1.26283 w_k1_KO_s007 + 1.75543 w_k1_MAT_s007
                                 + 0.991854 w_k1_MSFT_s007
                                 + 1.21719 w_k1_PG_s007 >= 1
 z_k1_s0076_constraint:          var_1 + z_k1_s0076 + 0.887559 w_k1_ADM_s007
                                 + 0.758236 w_k1_AXP_s007
                                 + 0.685339 w_k1_DIS_s007
                                 + 0.883535 w_k1_FDX_s007
                                 + 0.670692 w_k1_IBM_s007
                                 + 0.902952 w_k1_INTC_s007
                                 + 0.725129 w_k1_KO_s007
                                 + 0.621561 w_k1_MAT_s007
                                 + 0.862723 w_k1_MSFT_s007
                                 + 0.884543 w_k1_PG_s007 >= 1
 z_k1_s0077_constraint:          var_1 + z_k1_s0077 + 0.89798 w_k1_ADM_s007
                                 + 0.994541 w_k1_AXP_s007
                                 + 1.01764 w_k1_DIS_s007
                                 + 1.09792 w_k1_FDX_s007
                                 + 0.907781 w_k1_IBM_s007
                                 + 1.08155 w_k1_INTC_s007
                                 + 0.883586 w_k1_KO_s007
                                 + 0.77024 w_k1_MAT_s007
                                 + 0.965568 w_k1_MSFT_s007
                                 + 1.11631 w_k1_PG_s007 >= 1
 z_k1_s0100_constraint:          var_1 + z_k1_s0100 + 1.3125 w_k1_ADM_s010
                                 + 1.20343 w_k1_AXP_s010
                                 + 1.73524 w_k1_DIS_s010
                                 + 1.32837 w_k1_FDX_s010
                                 + 1.08931 w_k1_IBM_s010
                                 + 1.34004 w_k1_INTC_s010 + 1.3438 w_k1_KO_s010
                                 + 1.19082 w_k1_MAT_s010
                                 + 1.19633 w_k1_MSFT_s010
                                 + 1.17001 w_k1_PG_s010 >= 1
 z_k1_s0101_constraint:          var_1 + z_k1_s0101 + 0.743315 w_k1_ADM_s010
                                 + 1.27957 w_k1_AXP_s010
                                 + 1.61996 w_k1_DIS_s010 + 1.1153 w_k1_FDX_s010
                                 + 1.74652 w_k1_IBM_s010
                                 + 1.05757 w_k1_INTC_s010
                                 + 1.40381 w_k1_KO_s010 + 2.35181 w_k1_MAT_s010
                                 + 1.02474 w_k1_MSFT_s010
                                 + 1.21532 w_k1_PG_s010 >= 1
 z_k1_s0102_constraint:          var_1 + z_k1_s0102 + 1.04447 w_k1_ADM_s010
                                 + 0.963472 w_k1_AXP_s010
                                 + 1.29333 w_k1_DIS_s010
                                 + 0.994134 w_k1_FDX_s010
                                 + 1.11412 w_k1_IBM_s010
                                 + 0.930184 w_k1_INTC_s010
                                 + 1.32868 w_k1_KO_s010 + 1.36601 w_k1_MAT_s010
                                 + 0.982918 w_k1_MSFT_s010
                                 + 1.10328 w_k1_PG_s010 >= 1
 z_k1_s0103_constraint:          var_1 + z_k1_s0103 + 0.92146 w_k1_ADM_s010
                                 + 1.19439 w_k1_AXP_s010
                                 + 0.90021 w_k1_DIS_s010
                                 + 0.975747 w_k1_FDX_s010
                                 + 1.0337 w_k1_IBM_s010
                                 + 0.854544 w_k1_INTC_s010
                                 + 1.01116 w_k1_KO_s010 + 1.78484 w_k1_MAT_s010
                                 + 0.784306 w_k1_MSFT_s010
                                 + 1.01518 w_k1_PG_s010 >= 1
 z_k1_s0104_constraint:          var_1 + z_k1_s0104 + 1.03726 w_k1_ADM_s010
                                 + 1.13428 w_k1_AXP_s010
                                 + 0.961628 w_k1_DIS_s010
                                 + 0.924368 w_k1_FDX_s010
                                 + 1.37322 w_k1_IBM_s010
                                 + 0.836924 w_k1_INTC_s010
                                 + 1.11667 w_k1_KO_s010 + 1.63102 w_k1_MAT_s010
                                 + 0.812226 w_k1_MSFT_s010
                                 + 0.979234 w_k1_PG_s010 >= 1
 z_k1_s0105_constraint:          var_1 + z_k1_s0105 + 0.869485 w_k1_ADM_s010
                                 + 0.766021 w_k1_AXP_s010
                                 + 0.88537 w_k1_DIS_s010
                                 + 0.818909 w_k1_FDX_s010
                                 + 1.07018 w_k1_IBM_s010
                                 + 0.945968 w_k1_INTC_s010
                                 + 1.01283 w_k1_KO_s010 + 1.0743 w_k1_MAT_s010
                                 + 0.920745 w_k1_MSFT_s010
                                 + 0.978265 w_k1_PG_s010 >= 1
 z_k1_s0106_constraint:          var_1 + z_k1_s0106 + 0.824539 w_k1_ADM_s010
                                 + 1.13543 w_k1_AXP_s010
                                 + 0.841874 w_k1_DIS_s010
                                 + 0.946473 w_k1_FDX_s010
                                 + 0.819658 w_k1_IBM_s010
                                 + 0.78509 w_k1_INTC_s010
                                 + 0.750254 w_k1_KO_s010
                                 + 0.94321 w_k1_MAT_s010
                                 + 0.739821 w_k1_MSFT_s010
                                 + 0.968314 w_k1_PG_s010 >= 1
 z_k1_s0107_constraint:          var_1 + z_k1_s0107 + 0.815081 w_k1_ADM_s010
                                 + 0.734525 w_k1_AXP_s010
                                 + 0.90758 w_k1_DIS_s010
                                 + 0.984055 w_k1_FDX_s010
                                 + 0.771066 w_k1_IBM_s010
                                 + 0.976783 w_k1_INTC_s010
                                 + 0.806238 w_k1_KO_s010
                                 + 0.951194 w_k1_MAT_s010
                                 + 0.908163 w_k1_MSFT_s010
                                 + 0.893921 w_k1_PG_s010 >= 1
 z_k1_s0110_constraint:          var_1 + z_k1_s0110 + 1.53642 w_k1_ADM_s011
                                 + 1.32123 w_k1_AXP_s011
                                 + 1.27519 w_k1_DIS_s011
                                 + 1.30447 w_k1_FDX_s011
                                 + 0.933764 w_k1_IBM_s011
                                 + 1.29455 w_k1_INTC_s011
                                 + 1.36093 w_k1_KO_s011
                                 + 0.915602 w_k1_MAT_s011
                                 + 1.27309 w_k1_MSFT_s011
                                 + 1.21255 w_k1_PG_s011 >= 1
 z_k1_s0111_constraint:          var_1 + z_k1_s0111 + 1.20636 w_k1_ADM_s011
                                 + 1.00342 w_k1_AXP_s011
                                 + 1.00762 w_k1_DIS_s011
                                 + 1.05679 w_k1_FDX_s011
                                 + 0.97296 w_k1_IBM_s011
                                 + 0.860756 w_k1_INTC_s011
                                 + 1.03828 w_k1_KO_s011
                                 + 0.985719 w_k1_MAT_s011
                                 + 0.946288 w_k1_MSFT_s011
                                 + 1.13182 w_k1_PG_s011 >= 1
 z_k1_s0112_constraint:          var_1 + z_k1_s0112 + 1.12184 w_k1_ADM_s011
                                 + 0.887437 w_k1_AXP_s011
                                 + 1.18149 w_k1_DIS_s011
                                 + 0.989198 w_k1_FDX_s011
                                 + 1.20481 w_k1_IBM_s011
                                 + 1.03929 w_k1_INTC_s011
                                 + 1.11747 w_k1_KO_s011 + 1.12648 w_k1_MAT_s011
                                 + 1.04509 w_k1_MSFT_s011 + 1.2106 w_k1_PG_s011
                                 >= 1
 z_k1_s0113_constraint:          var_1 + z_k1_s0113 + 0.991068 w_k1_ADM_s011
                                 + 1.52587 w_k1_AXP_s011
                                 + 1.33176 w_k1_DIS_s011
                                 + 1.22295 w_k1_FDX_s011
                                 + 1.52425 w_k1_IBM_s011
                                 + 1.27688 w_k1_INTC_s011
                                 + 1.33307 w_k1_KO_s011 + 1.81684 w_k1_MAT_s011
                                 + 1.42582 w_k1_MSFT_s011
                                 + 1.33576 w_k1_PG_s011 >= 1
 z_k1_s0114_constraint:          var_1 + z_k1_s0114 + 0.905963 w_k1_ADM_s011
                                 + 0.709846 w_k1_AXP_s011
                                 + 0.554818 w_k1_DIS_s011
                                 + 0.69047 w_k1_FDX_s011
                                 + 0.907489 w_k1_IBM_s011
                                 + 0.825507 w_k1_INTC_s011
                                 + 0.871472 w_k1_KO_s011
                                 + 0.710514 w_k1_MAT_s011
                                 + 0.810726 w_k1_MSFT_s011
                                 + 0.847526 w_k1_PG_s011 >= 1
 z_k1_s0115_constraint:          var_1 + z_k1_s0115 + 1.10273 w_k1_ADM_s011
                                 + 0.793025 w_k1_AXP_s011
                                 + 1.34295 w_k1_DIS_s011
                                 + 1.13245 w_k1_FDX_s011
                                 + 0.823669 w_k1_IBM_s011
                                 + 0.869701 w_k1_INTC_s011
                                 + 1.04797 w_k1_KO_s011
                                 + 0.563764 w_k1_MAT_s011
                                 + 1.21293 w_k1_MSFT_s011
                                 + 1.00292 w_k1_PG_s011 >= 1
 z_k1_s0116_constraint:          var_1 + z_k1_s0116 + 0.759547 w_k1_ADM_s011
                                 + 0.946552 w_k1_AXP_s011
                                 + 0.708875 w_k1_DIS_s011
                                 + 0.757153 w_k1_FDX_s011
                                 + 1.25178 w_k1_IBM_s011
                                 + 0.857486 w_k1_INTC_s011
                                 + 0.948273 w_k1_KO_s011
                                 + 1.13028 w_k1_MAT_s011
                                 + 0.876791 w_k1_MSFT_s011
                                 + 0.947059 w_k1_PG_s011 >= 1
 z_k1_s0117_constraint:          var_1 + z_k1_s0117 + 0.905064 w_k1_ADM_s011
                                 + 0.851262 w_k1_AXP_s011
                                 + 0.67665 w_k1_DIS_s011
                                 + 0.838097 w_k1_FDX_s011
                                 + 0.708311 w_k1_IBM_s011
                                 + 0.641391 w_k1_INTC_s011
                                 + 0.757436 w_k1_KO_s011
                                 + 0.761447 w_k1_MAT_s011
                                 + 0.849089 w_k1_MSFT_s011
                                 + 0.927232 w_k1_PG_s011 >= 1
 z_k1_s0120_constraint:          var_1 + z_k1_s0120 + 0.683185 w_k1_ADM_s012
                                 + 0.666069 w_k1_AXP_s012
                                 + 0.661099 w_k1_DIS_s012
                                 + 0.755458 w_k1_FDX_s012
                                 + 0.775099 w_k1_IBM_s012
                                 + 0.741623 w_k1_INTC_s012
                                 + 0.769195 w_k1_KO_s012
                                 + 0.648226 w_k1_MAT_s012
                                 + 0.775203 w_k1_MSFT_s012
                                 + 0.904957 w_k1_PG_s012 >= 1
 z_k1_s0121_constraint:          var_1 + z_k1_s0121 + 1.11587 w_k1_ADM_s012
                                 + 1.05458 w_k1_AXP_s012
                                 + 1.26083 w_k1_DIS_s012
                                 + 1.19834 w_k1_FDX_s012
                                 + 0.878782 w_k1_IBM_s012
                                 + 0.905608 w_k1_INTC_s012
                                 + 0.996951 w_k1_KO_s012
                                 + 0.931805 w_k1_MAT_s012
                                 + 1.08368 w_k1_MSFT_s012
                                 + 1.11822 w_k1_PG_s012 >= 1
 z_k1_s0122_constraint:          var_1 + z_k1_s0122 + 0.921444 w_k1_ADM_s012
                                 + 1.17522 w_k1_AXP_s012
                                 + 1.38903 w_k1_DIS_s012
                                 + 1.17704 w_k1_FDX_s012
                                 + 1.73781 w_k1_IBM_s012
                                 + 0.924609 w_k1_INTC_s012
                                 + 1.26216 w_k1_KO_s012 + 1.39477 w_k1_MAT_s012
                                 + 1.04281 w_k1_MSFT_s012
                                 + 1.17026 w_k1_PG_s012 >= 1
 z_k1_s0123_constraint:          var_1 + z_k1_s0123 + 0.692136 w_k1_ADM_s012
                                 + 0.897154 w_k1_AXP_s012
                                 + 0.746598 w_k1_DIS_s012
                                 + 0.87403 w_k1_FDX_s012 + 1.1034 w_k1_IBM_s012
                                 + 0.897656 w_k1_INTC_s012
                                 + 0.97381 w_k1_KO_s012 + 1.04027 w_k1_MAT_s012
                                 + 0.85192 w_k1_MSFT_s012 + 1.0929 w_k1_PG_s012
                                 >= 1
 z_k1_s0124_constraint:          var_1 + z_k1_s0124 + 1.11435 w_k1_ADM_s012
                                 + 1.04106 w_k1_AXP_s012
                                 + 0.815516 w_k1_DIS_s012
                                 + 1.01224 w_k1_FDX_s012
                                 + 0.749371 w_k1_IBM_s012
                                 + 0.781671 w_k1_INTC_s012
                                 + 0.874796 w_k1_KO_s012
                                 + 0.937722 w_k1_MAT_s012
                                 + 0.934315 w_k1_MSFT_s012
                                 + 1.0526 w_k1_PG_s012 >= 1
 z_k1_s0125_constraint:          var_1 + z_k1_s0125 + 1.20034 w_k1_ADM_s012
                                 + 1.48436 w_k1_AXP_s012
                                 + 1.05538 w_k1_DIS_s012
                                 + 1.13306 w_k1_FDX_s012
                                 + 1.30528 w_k1_IBM_s012
                                 + 0.887581 w_k1_INTC_s012 + 1.129 w_k1_KO_s012
                                 + 1.33528 w_k1_MAT_s012
                                 + 0.981567 w_k1_MSFT_s012
                                 + 1.16898 w_k1_PG_s012 >= 1
 z_k1_s0126_constraint:          var_1 + z_k1_s0126 + 0.963489 w_k1_ADM_s012
                                 + 0.964126 w_k1_AXP_s012
                                 + 1.19585 w_k1_DIS_s012
                                 + 1.07155 w_k1_FDX_s012
                                 + 1.24687 w_k1_IBM_s012
                                 + 0.934551 w_k1_INTC_s012
                                 + 1.28125 w_k1_KO_s012 + 1.19021 w_k1_MAT_s012
                                 + 1.17363 w_k1_MSFT_s012
                                 + 1.25237 w_k1_PG_s012 >= 1
 z_k1_s0127_constraint:          var_1 + z_k1_s0127 + 1.17893 w_k1_ADM_s012
                                 + 1.36091 w_k1_AXP_s012
                                 + 1.75008 w_k1_DIS_s012 + 1.3691 w_k1_FDX_s012
                                 + 1.14796 w_k1_IBM_s012
                                 + 1.15373 w_k1_INTC_s012
                                 + 1.21769 w_k1_KO_s012 + 1.12115 w_k1_MAT_s012
                                 + 1.26689 w_k1_MSFT_s012
                                 + 1.35919 w_k1_PG_s012 >= 1
 z_k1_s0130_constraint:          var_1 + z_k1_s0130 + 0.978845 w_k1_ADM_s013
                                 + 0.703891 w_k1_AXP_s013
                                 + 0.792357 w_k1_DIS_s013
                                 + 0.815567 w_k1_FDX_s013
                                 + 0.859247 w_k1_IBM_s013
                                 + 0.730027 w_k1_INTC_s013
                                 + 0.806559 w_k1_KO_s013
                                 + 0.834459 w_k1_MAT_s013
                                 + 0.8546 w_k1_MSFT_s013 + 1.0125 w_k1_PG_s013
                                 >= 1
 z_k1_s0131_constraint:          var_1 + z_k1_s0131 + 0.864299 w_k1_ADM_s013
                                 + 0.710726 w_k1_AXP_s013
                                 + 0.755693 w_k1_DIS_s013
                                 + 0.848401 w_k1_FDX_s013
                                 + 0.656134 w_k1_IBM_s013
                                 + 0.756787 w_k1_INTC_s013
                                 + 0.65357 w_k1_KO_s013
                                 + 0.579341 w_k1_MAT_s013
                                 + 0.875468 w_k1_MSFT_s013
                                 + 0.903424 w_k1_PG_s013 >= 1
 z_k1_s0132_constraint:          var_1 + z_k1_s0132 + 1.00821 w_k1_ADM_s013
                                 + 1.18235 w_k1_AXP_s013
                                 + 1.60487 w_k1_DIS_s013
                                 + 1.11293 w_k1_FDX_s013
                                 + 1.73255 w_k1_IBM_s013
                                 + 1.23575 w_k1_INTC_s013
                                 + 1.40237 w_k1_KO_s013 + 1.48686 w_k1_MAT_s013
                                 + 1.39364 w_k1_MSFT_s013
                                 + 1.15887 w_k1_PG_s013 >= 1
 z_k1_s0133_constraint:          var_1 + z_k1_s0133 + 1.33581 w_k1_ADM_s013
                                 + 1.33523 w_k1_AXP_s013 + 1.7108 w_k1_DIS_s013
                                 + 1.31828 w_k1_FDX_s013
                                 + 1.14514 w_k1_IBM_s013
                                 + 1.40801 w_k1_INTC_s013
                                 + 1.38104 w_k1_KO_s013 + 1.36596 w_k1_MAT_s013
                                 + 1.44721 w_k1_MSFT_s013 + 1.2832 w_k1_PG_s013
                                 >= 1
 z_k1_s0134_constraint:          var_1 + z_k1_s0134 + 0.761485 w_k1_ADM_s013
                                 + 0.6681 w_k1_AXP_s013
                                 + 0.755404 w_k1_DIS_s013
                                 + 0.639929 w_k1_FDX_s013
                                 + 1.20306 w_k1_IBM_s013
                                 + 0.831176 w_k1_INTC_s013
                                 + 0.924625 w_k1_KO_s013
                                 + 1.37207 w_k1_MAT_s013
                                 + 0.836754 w_k1_MSFT_s013
                                 + 0.835686 w_k1_PG_s013 >= 1
 z_k1_s0135_constraint:          var_1 + z_k1_s0135 + 1.1326 w_k1_ADM_s013
                                 + 0.915673 w_k1_AXP_s013
                                 + 1.06236 w_k1_DIS_s013
                                 + 0.880913 w_k1_FDX_s013
                                 + 1.11905 w_k1_IBM_s013
                                 + 0.958903 w_k1_INTC_s013
                                 + 1.1245 w_k1_KO_s013 + 1.37813 w_k1_MAT_s013
                                 + 1.05261 w_k1_MSFT_s013
                                 + 1.09654 w_k1_PG_s013 >= 1
 z_k1_s0136_constraint:          var_1 + z_k1_s0136 + 1.45972 w_k1_ADM_s013
                                 + 0.983015 w_k1_AXP_s013
                                 + 1.37363 w_k1_DIS_s013
                                 + 1.03384 w_k1_FDX_s013
                                 + 1.12444 w_k1_IBM_s013
                                 + 0.957653 w_k1_INTC_s013
                                 + 1.16325 w_k1_KO_s013 + 1.03897 w_k1_MAT_s013
                                 + 1.10722 w_k1_MSFT_s013
                                 + 1.21068 w_k1_PG_s013 >= 1
 z_k1_s0137_constraint:          var_1 + z_k1_s0137 + 1.44409 w_k1_ADM_s013
                                 + 1.1438 w_k1_AXP_s013 + 1.21451 w_k1_DIS_s013
                                 + 1.31567 w_k1_FDX_s013
                                 + 0.84218 w_k1_IBM_s013
                                 + 0.997454 w_k1_INTC_s013
                                 + 0.973435 w_k1_KO_s013
                                 + 1.29956 w_k1_MAT_s013
                                 + 1.17067 w_k1_MSFT_s013
                                 + 1.26866 w_k1_PG_s013 >= 1
 z_k1_s0140_constraint:          var_1 + z_k1_s0140 + 1.37885 w_k1_ADM_s014
                                 + 1.41581 w_k1_AXP_s014
                                 + 1.26398 w_k1_DIS_s014
                                 + 1.37604 w_k1_FDX_s014
                                 + 1.05439 w_k1_IBM_s014
                                 + 1.27239 w_k1_INTC_s014
                                 + 1.15103 w_k1_KO_s014 + 1.10367 w_k1_MAT_s014
                                 + 1.25931 w_k1_MSFT_s014
                                 + 1.29021 w_k1_PG_s014 >= 1
 z_k1_s0141_constraint:          var_1 + z_k1_s0141 + 1.17237 w_k1_ADM_s014
                                 + 1.2332 w_k1_AXP_s014
                                 + 0.797047 w_k1_DIS_s014
                                 + 0.989939 w_k1_FDX_s014
                                 + 0.907768 w_k1_IBM_s014
                                 + 0.77038 w_k1_INTC_s014
                                 + 0.804712 w_k1_KO_s014
                                 + 1.25307 w_k1_MAT_s014
                                 + 0.849573 w_k1_MSFT_s014
                                 + 1.01393 w_k1_PG_s014 >= 1
 z_k1_s0142_constraint:          var_1 + z_k1_s0142 + 1.08492 w_k1_ADM_s014
                                 + 1.19934 w_k1_AXP_s014
                                 + 1.07449 w_k1_DIS_s014
                                 + 1.08597 w_k1_FDX_s014
                                 + 1.36524 w_k1_IBM_s014
                                 + 1.04544 w_k1_INTC_s014
                                 + 1.27079 w_k1_KO_s014 + 1.23295 w_k1_MAT_s014
                                 + 1.04328 w_k1_MSFT_s014
                                 + 1.13139 w_k1_PG_s014 >= 1
 z_k1_s0143_constraint:          var_1 + z_k1_s0143 + 1.0507 w_k1_ADM_s014
                                 + 1.01403 w_k1_AXP_s014
                                 + 0.955416 w_k1_DIS_s014
                                 + 1.04343 w_k1_FDX_s014
                                 + 0.669795 w_k1_IBM_s014
                                 + 0.886022 w_k1_INTC_s014
                                 + 0.843261 w_k1_KO_s014
                                 + 0.587867 w_k1_MAT_s014
                                 + 1.00399 w_k1_MSFT_s014
                                 + 0.897044 w_k1_PG_s014 >= 1
 z_k1_s0144_constraint:          var_1 + z_k1_s0144 + 1.1266 w_k1_ADM_s014
                                 + 1.09274 w_k1_AXP_s014
                                 + 1.40964 w_k1_DIS_s014
                                 + 1.38769 w_k1_FDX_s014
                                 + 0.922987 w_k1_IBM_s014
                                 + 1.17522 w_k1_INTC_s014
                                 + 1.02586 w_k1_KO_s014
                                 + 0.902978 w_k1_MAT_s014
                                 + 1.11973 w_k1_MSFT_s014
                                 + 1.12598 w_k1_PG_s014 >= 1
 z_k1_s0145_constraint:          var_1 + z_k1_s0145 + 0.835624 w_k1_ADM_s014
                                 + 0.848278 w_k1_AXP_s014
                                 + 0.93342 w_k1_DIS_s014
                                 + 1.00599 w_k1_FDX_s014
                                 + 1.00083 w_k1_IBM_s014
                                 + 1.01068 w_k1_INTC_s014
                                 + 1.04611 w_k1_KO_s014 + 1.03193 w_k1_MAT_s014
                                 + 0.926261 w_k1_MSFT_s014
                                 + 0.996276 w_k1_PG_s014 >= 1
 z_k1_s0146_constraint:          var_1 + z_k1_s0146 + 0.932515 w_k1_ADM_s014
                                 + 0.730546 w_k1_AXP_s014
                                 + 0.659337 w_k1_DIS_s014
                                 + 0.799729 w_k1_FDX_s014
                                 + 0.639864 w_k1_IBM_s014
                                 + 0.776958 w_k1_INTC_s014
                                 + 0.712881 w_k1_KO_s014
                                 + 0.473974 w_k1_MAT_s014
                                 + 0.825163 w_k1_MSFT_s014
                                 + 0.80098 w_k1_PG_s014 >= 1
 z_k1_s0147_constraint:          var_1 + z_k1_s0147 + 0.842047 w_k1_ADM_s014
                                 + 0.557919 w_k1_AXP_s014
                                 + 0.565331 w_k1_DIS_s014
                                 + 0.674306 w_k1_FDX_s014
                                 + 0.922301 w_k1_IBM_s014
                                 + 0.786729 w_k1_INTC_s014
                                 + 0.840042 w_k1_KO_s014
                                 + 0.840557 w_k1_MAT_s014
                                 + 0.737607 w_k1_MSFT_s014
                                 + 0.810824 w_k1_PG_s014 >= 1
 z_k1_s0150_constraint:          var_1 + z_k1_s0150 + 0.962178 w_k1_ADM_s015
                                 + 0.761008 w_k1_AXP_s015
                                 + 0.632832 w_k1_DIS_s015
                                 + 0.808847 w_k1_FDX_s015
                                 + 0.686776 w_k1_IBM_s015
                                 + 1.12597 w_k1_INTC_s015
                                 + 0.668699 w_k1_KO_s015
                                 + 0.596936 w_k1_MAT_s015
                                 + 0.722537 w_k1_MSFT_s015
                                 + 0.828931 w_k1_PG_s015 >= 1
 z_k1_s0151_constraint:          var_1 + z_k1_s0151 + 1.06122 w_k1_ADM_s015
                                 + 1.08094 w_k1_AXP_s015
                                 + 1.04663 w_k1_DIS_s015
                                 + 1.01304 w_k1_FDX_s015
                                 + 1.18335 w_k1_IBM_s015
                                 + 1.10714 w_k1_INTC_s015
                                 + 1.26331 w_k1_KO_s015 + 1.27899 w_k1_MAT_s015
                                 + 1.04014 w_k1_MSFT_s015
                                 + 1.14894 w_k1_PG_s015 >= 1
 z_k1_s0152_constraint:          var_1 + z_k1_s0152 + 1.53969 w_k1_ADM_s015
                                 + 1.23972 w_k1_AXP_s015
                                 + 1.40272 w_k1_DIS_s015
                                 + 1.39989 w_k1_FDX_s015
                                 + 0.839975 w_k1_IBM_s015
                                 + 1.42292 w_k1_INTC_s015
                                 + 1.17995 w_k1_KO_s015
                                 + 0.908621 w_k1_MAT_s015
                                 + 1.32481 w_k1_MSFT_s015
                                 + 1.12358 w_k1_PG_s015 >= 1
 z_k1_s0153_constraint:          var_1 + z_k1_s0153 + 1.12311 w_k1_ADM_s015
                                 + 1.09938 w_k1_AXP_s015
                                 + 0.815529 w_k1_DIS_s015
                                 + 0.917547 w_k1_FDX_s015
                                 + 0.866075 w_k1_IBM_s015
                                 + 1.16794 w_k1_INTC_s015
                                 + 0.96716 w_k1_KO_s015
                                 + 0.999181 w_k1_MAT_s015
                                 + 0.919352 w_k1_MSFT_s015
                                 + 0.878185 w_k1_PG_s015 >= 1
 z_k1_s0154_constraint:          var_1 + z_k1_s0154 + 1.16679 w_k1_ADM_s015
                                 + 1.61457 w_k1_AXP_s015 + 1.4383 w_k1_DIS_s015
                                 + 1.24594 w_k1_FDX_s015 + 1.1296 w_k1_IBM_s015
                                 + 1.61001 w_k1_INTC_s015
                                 + 1.28497 w_k1_KO_s015 + 1.0454 w_k1_MAT_s015
                                 + 1.19548 w_k1_MSFT_s015 + 1.1217 w_k1_PG_s015
                                 >= 1
 z_k1_s0155_constraint:          var_1 + z_k1_s0155 + 1.31407 w_k1_ADM_s015
                                 + 1.3557 w_k1_AXP_s015 + 1.54249 w_k1_DIS_s015
                                 + 1.2122 w_k1_FDX_s015 + 1.66581 w_k1_IBM_s015
                                 + 1.4814 w_k1_INTC_s015 + 1.63517 w_k1_KO_s015
                                 + 1.56088 w_k1_MAT_s015
                                 + 1.30698 w_k1_MSFT_s015 + 1.1578 w_k1_PG_s015
                                 >= 1
 z_k1_s0156_constraint:          var_1 + z_k1_s0156 + 0.967422 w_k1_ADM_s015
                                 + 0.873085 w_k1_AXP_s015
                                 + 0.970548 w_k1_DIS_s015
                                 + 0.903657 w_k1_FDX_s015
                                 + 0.793013 w_k1_IBM_s015
                                 + 0.995467 w_k1_INTC_s015
                                 + 0.995221 w_k1_KO_s015
                                 + 0.788063 w_k1_MAT_s015
                                 + 0.971507 w_k1_MSFT_s015
                                 + 0.922473 w_k1_PG_s015 >= 1
 z_k1_s0157_constraint:          var_1 + z_k1_s0157 + 1.01899 w_k1_ADM_s015
                                 + 1.14944 w_k1_AXP_s015
                                 + 1.08929 w_k1_DIS_s015
                                 + 1.10397 w_k1_FDX_s015
                                 + 0.838796 w_k1_IBM_s015
                                 + 1.32081 w_k1_INTC_s015
                                 + 1.13873 w_k1_KO_s015
                                 + 0.990851 w_k1_MAT_s015
                                 + 1.09893 w_k1_MSFT_s015
                                 + 1.07611 w_k1_PG_s015 >= 1
 z_k1_s0160_constraint:          var_1 + z_k1_s0160 + 1.343 w_k1_ADM_s016
                                 + 1.20242 w_k1_AXP_s016
                                 + 1.00961 w_k1_DIS_s016
                                 + 0.901817 w_k1_FDX_s016
                                 + 1.39503 w_k1_IBM_s016
                                 + 1.23716 w_k1_INTC_s016
                                 + 1.24269 w_k1_KO_s016
                                 + 0.910127 w_k1_MAT_s016
                                 + 0.973221 w_k1_MSFT_s016
                                 + 1.07044 w_k1_PG_s016 >= 1
 z_k1_s0161_constraint:          var_1 + z_k1_s0161 + 1.20565 w_k1_ADM_s016
                                 + 0.954015 w_k1_AXP_s016
                                 + 0.870884 w_k1_DIS_s016
                                 + 1.07513 w_k1_FDX_s016 + 0.7728 w_k1_IBM_s016
                                 + 1.20855 w_k1_INTC_s016
                                 + 0.907428 w_k1_KO_s016
                                 + 0.541503 w_k1_MAT_s016
                                 + 0.972326 w_k1_MSFT_s016
                                 + 0.933969 w_k1_PG_s016 >= 1
 z_k1_s0162_constraint:          var_1 + z_k1_s0162 + 1.30617 w_k1_ADM_s016
                                 + 1.84012 w_k1_AXP_s016 + 1.616 w_k1_DIS_s016
                                 + 1.51014 w_k1_FDX_s016
                                 + 1.59003 w_k1_IBM_s016
                                 + 1.53217 w_k1_INTC_s016
                                 + 1.62034 w_k1_KO_s016 + 2.05913 w_k1_MAT_s016
                                 + 1.26039 w_k1_MSFT_s016
                                 + 1.43748 w_k1_PG_s016 >= 1
 z_k1_s0163_constraint:          var_1 + z_k1_s0163 + 1.20897 w_k1_ADM_s016
                                 + 1.2186 w_k1_AXP_s016 + 1.88922 w_k1_DIS_s016
                                 + 1.41035 w_k1_FDX_s016
                                 + 1.06038 w_k1_IBM_s016
                                 + 1.27979 w_k1_INTC_s016
                                 + 1.38865 w_k1_KO_s016 + 1.11276 w_k1_MAT_s016
                                 + 1.36582 w_k1_MSFT_s016
                                 + 1.26878 w_k1_PG_s016 >= 1
 z_k1_s0164_constraint:          var_1 + z_k1_s0164 + 1.11783 w_k1_ADM_s016
                                 + 0.816461 w_k1_AXP_s016
                                 + 0.602166 w_k1_DIS_s016
                                 + 0.81352 w_k1_FDX_s016
                                 + 0.553335 w_k1_IBM_s016
                                 + 0.734963 w_k1_INTC_s016
                                 + 0.745942 w_k1_KO_s016
                                 + 0.517847 w_k1_MAT_s016
                                 + 0.806946 w_k1_MSFT_s016
                                 + 0.915636 w_k1_PG_s016 >= 1
 z_k1_s0165_constraint:          var_1 + z_k1_s0165 + 1.29028 w_k1_ADM_s016
                                 + 1.24153 w_k1_AXP_s016
                                 + 1.11054 w_k1_DIS_s016
                                 + 1.13371 w_k1_FDX_s016
                                 + 0.963617 w_k1_IBM_s016
                                 + 1.13597 w_k1_INTC_s016
                                 + 1.10189 w_k1_KO_s016 + 1.07042 w_k1_MAT_s016
                                 + 1.05951 w_k1_MSFT_s016
                                 + 1.15919 w_k1_PG_s016 >= 1
 z_k1_s0166_constraint:          var_1 + z_k1_s0166 + 1.52369 w_k1_ADM_s016
                                 + 1.36461 w_k1_AXP_s016
                                 + 1.49405 w_k1_DIS_s016
                                 + 1.40134 w_k1_FDX_s016
                                 + 0.694969 w_k1_IBM_s016
                                 + 1.2795 w_k1_INTC_s016 + 1.02626 w_k1_KO_s016
                                 + 0.760813 w_k1_MAT_s016
                                 + 1.21452 w_k1_MSFT_s016
                                 + 1.22655 w_k1_PG_s016 >= 1
 z_k1_s0167_constraint:          var_1 + z_k1_s0167 + 0.855707 w_k1_ADM_s016
                                 + 0.900475 w_k1_AXP_s016
                                 + 0.735175 w_k1_DIS_s016
                                 + 0.786277 w_k1_FDX_s016
                                 + 0.976544 w_k1_IBM_s016
                                 + 0.7104 w_k1_INTC_s016
                                 + 0.864225 w_k1_KO_s016
                                 + 0.925205 w_k1_MAT_s016
                                 + 0.655551 w_k1_MSFT_s016
                                 + 0.974814 w_k1_PG_s016 >= 1
 z_k1_s0170_constraint:          var_1 + z_k1_s0170 + 0.984644 w_k1_ADM_s017
                                 + 0.750494 w_k1_AXP_s017
                                 + 1.20757 w_k1_DIS_s017
                                 + 0.891477 w_k1_FDX_s017
                                 + 0.959607 w_k1_IBM_s017
                                 + 0.848323 w_k1_INTC_s017
                                 + 1.06845 w_k1_KO_s017
                                 + 0.965018 w_k1_MAT_s017
                                 + 1.16535 w_k1_MSFT_s017
                                 + 0.961028 w_k1_PG_s017 >= 1
 z_k1_s0171_constraint:          var_1 + z_k1_s0171 + 1.17603 w_k1_ADM_s017
                                 + 1.41469 w_k1_AXP_s017
                                 + 1.24371 w_k1_DIS_s017
                                 + 1.10201 w_k1_FDX_s017
                                 + 1.50278 w_k1_IBM_s017
                                 + 0.859355 w_k1_INTC_s017
                                 + 1.16134 w_k1_KO_s017 + 1.57828 w_k1_MAT_s017
                                 + 1.11213 w_k1_MSFT_s017 + 1.1744 w_k1_PG_s017
                                 >= 1
 z_k1_s0172_constraint:          var_1 + z_k1_s0172 + 0.645623 w_k1_ADM_s017
                                 + 0.822231 w_k1_AXP_s017
                                 + 0.862641 w_k1_DIS_s017
                                 + 1.00237 w_k1_FDX_s017
                                 + 0.726105 w_k1_IBM_s017
                                 + 0.988764 w_k1_INTC_s017
                                 + 0.769021 w_k1_KO_s017
                                 + 0.718228 w_k1_MAT_s017
                                 + 1.01186 w_k1_MSFT_s017
                                 + 0.876018 w_k1_PG_s017 >= 1
 z_k1_s0173_constraint:          var_1 + z_k1_s0173 + 0.988338 w_k1_ADM_s017
                                 + 1.22391 w_k1_AXP_s017
                                 + 1.30462 w_k1_DIS_s017
                                 + 1.28206 w_k1_FDX_s017
                                 + 0.923558 w_k1_IBM_s017
                                 + 1.19196 w_k1_INTC_s017
                                 + 0.966385 w_k1_KO_s017
                                 + 1.05978 w_k1_MAT_s017
                                 + 1.35608 w_k1_MSFT_s017
                                 + 1.10212 w_k1_PG_s017 >= 1
 z_k1_s0174_constraint:          var_1 + z_k1_s0174 + 0.859447 w_k1_ADM_s017
                                 + 1.02507 w_k1_AXP_s017
                                 + 0.876865 w_k1_DIS_s017
                                 + 0.946562 w_k1_FDX_s017
                                 + 0.91282 w_k1_IBM_s017
                                 + 1.07615 w_k1_INTC_s017
                                 + 0.76637 w_k1_KO_s017
                                 + 0.711746 w_k1_MAT_s017
                                 + 1.03261 w_k1_MSFT_s017
                                 + 0.892761 w_k1_PG_s017 >= 1
 z_k1_s0175_constraint:          var_1 + z_k1_s0175 + 0.639375 w_k1_ADM_s017
                                 + 0.60207 w_k1_AXP_s017
                                 + 0.767159 w_k1_DIS_s017
                                 + 0.779672 w_k1_FDX_s017
                                 + 0.691211 w_k1_IBM_s017
                                 + 0.731113 w_k1_INTC_s017
                                 + 0.773027 w_k1_KO_s017
                                 + 0.80839 w_k1_MAT_s017
                                 + 0.954259 w_k1_MSFT_s017
                                 + 0.795126 w_k1_PG_s017 >= 1
 z_k1_s0176_constraint:          var_1 + z_k1_s0176 + 0.918781 w_k1_ADM_s017
                                 + 1.20865 w_k1_AXP_s017
                                 + 1.35176 w_k1_DIS_s017
                                 + 1.05388 w_k1_FDX_s017
                                 + 1.20999 w_k1_IBM_s017
                                 + 1.13882 w_k1_INTC_s017
                                 + 1.03404 w_k1_KO_s017
                                 + 0.938003 w_k1_MAT_s017
                                 + 1.2834 w_k1_MSFT_s017
                                 + 0.972158 w_k1_PG_s017 >= 1
 z_k1_s0177_constraint:          var_1 + z_k1_s0177 + 0.663491 w_k1_ADM_s017
                                 + 0.870983 w_k1_AXP_s017
                                 + 0.711802 w_k1_DIS_s017
                                 + 0.795045 w_k1_FDX_s017
                                 + 1.12631 w_k1_IBM_s017
                                 + 0.845792 w_k1_INTC_s017
                                 + 0.818942 w_k1_KO_s017
                                 + 1.32154 w_k1_MAT_s017
                                 + 0.882947 w_k1_MSFT_s017
                                 + 0.816447 w_k1_PG_s017 >= 1
 z_k1_s1000_constraint:          var_1 + z_k1_s1000 + 1.12808 w_k1_ADM_s100
                                 + 1.23786 w_k1_AXP_s100
                                 + 0.976684 w_k1_DIS_s100
                                 + 1.07033 w_k1_FDX_s100
                                 + 1.25017 w_k1_IBM_s100
                                 + 1.09506 w_k1_INTC_s100
                                 + 1.12193 w_k1_KO_s100 + 1.72791 w_k1_MAT_s100
                                 + 1.0624 w_k1_MSFT_s100 + 1.10943 w_k1_PG_s100
                                 >= 1
 z_k1_s1001_constraint:          var_1 + z_k1_s1001 + 1.32529 w_k1_ADM_s100
                                 + 1.37125 w_k1_AXP_s100 + 1.5076 w_k1_DIS_s100
                                 + 1.39216 w_k1_FDX_s100
                                 + 1.00672 w_k1_IBM_s100
                                 + 1.21832 w_k1_INTC_s100 + 1.1686 w_k1_KO_s100
                                 + 1.0949 w_k1_MAT_s100
                                 + 1.21258 w_k1_MSFT_s100
                                 + 1.13644 w_k1_PG_s100 >= 1
 z_k1_s1002_constraint:          var_1 + z_k1_s1002 + 0.706697 w_k1_ADM_s100
                                 + 0.626548 w_k1_AXP_s100
                                 + 0.64519 w_k1_DIS_s100
                                 + 0.761161 w_k1_FDX_s100
                                 + 0.748066 w_k1_IBM_s100
                                 + 0.7546 w_k1_INTC_s100
                                 + 0.661085 w_k1_KO_s100
                                 + 0.683314 w_k1_MAT_s100
                                 + 0.68675 w_k1_MSFT_s100
                                 + 0.782872 w_k1_PG_s100 >= 1
 z_k1_s1003_constraint:          var_1 + z_k1_s1003 + 1.02151 w_k1_ADM_s100
                                 + 0.822703 w_k1_AXP_s100
                                 + 1.00255 w_k1_DIS_s100
                                 + 0.890035 w_k1_FDX_s100
                                 + 1.0387 w_k1_IBM_s100
                                 + 0.873795 w_k1_INTC_s100
                                 + 1.0891 w_k1_KO_s100 + 1.2576 w_k1_MAT_s100
                                 + 1.03588 w_k1_MSFT_s100
                                 + 1.00015 w_k1_PG_s100 >= 1
 z_k1_s1004_constraint:          var_1 + z_k1_s1004 + 1.26195 w_k1_ADM_s100
                                 + 1.54445 w_k1_AXP_s100
                                 + 1.60769 w_k1_DIS_s100
                                 + 1.20772 w_k1_FDX_s100
                                 + 1.62839 w_k1_IBM_s100
                                 + 1.03859 w_k1_INTC_s100
                                 + 1.54391 w_k1_KO_s100 + 2.05044 w_k1_MAT_s100
                                 + 1.12905 w_k1_MSFT_s100
                                 + 1.26115 w_k1_PG_s100 >= 1
 z_k1_s1005_constraint:          var_1 + z_k1_s1005 + 0.905709 w_k1_ADM_s100
                                 + 0.813565 w_k1_AXP_s100
                                 + 0.885439 w_k1_DIS_s100
                                 + 0.962064 w_k1_FDX_s100
                                 + 0.747987 w_k1_IBM_s100
                                 + 0.936425 w_k1_INTC_s100
                                 + 0.823946 w_k1_KO_s100
                                 + 0.926868 w_k1_MAT_s100
                                 + 1.02612 w_k1_MSFT_s100
                                 + 0.970196 w_k1_PG_s100 >= 1
 z_k1_s1006_constraint:          var_1 + z_k1_s1006 + 1.50464 w_k1_ADM_s100
                                 + 1.25022 w_k1_AXP_s100 + 1.6066 w_k1_DIS_s100
                                 + 1.33653 w_k1_FDX_s100
                                 + 1.31234 w_k1_IBM_s100
                                 + 1.06352 w_k1_INTC_s100
                                 + 1.28153 w_k1_KO_s100 + 1.6076 w_k1_MAT_s100
                                 + 1.21672 w_k1_MSFT_s100
                                 + 1.22361 w_k1_PG_s100 >= 1
 z_k1_s1007_constraint:          var_1 + z_k1_s1007 + 1.06873 w_k1_ADM_s100
                                 + 1.07148 w_k1_AXP_s100
                                 + 1.12501 w_k1_DIS_s100
                                 + 1.05681 w_k1_FDX_s100
                                 + 1.01882 w_k1_IBM_s100
                                 + 1.07634 w_k1_INTC_s100
                                 + 1.06599 w_k1_KO_s100 + 1.10166 w_k1_MAT_s100
                                 + 1.17054 w_k1_MSFT_s100
                                 + 1.06671 w_k1_PG_s100 >= 1
 z_k1_s1010_constraint:          var_1 + z_k1_s1010 + 0.924022 w_k1_ADM_s101
                                 + 0.76621 w_k1_AXP_s101
                                 + 0.868099 w_k1_DIS_s101
                                 + 0.92528 w_k1_FDX_s101
                                 + 0.81655 w_k1_IBM_s101
                                 + 0.91531 w_k1_INTC_s101
                                 + 0.909962 w_k1_KO_s101
                                 + 0.568074 w_k1_MAT_s101
                                 + 0.986446 w_k1_MSFT_s101
                                 + 0.867373 w_k1_PG_s101 >= 1
 z_k1_s1011_constraint:          var_1 + z_k1_s1011 + 0.87942 w_k1_ADM_s101
                                 + 0.875716 w_k1_AXP_s101
                                 + 1.1932 w_k1_DIS_s101 + 1.0587 w_k1_FDX_s101
                                 + 1.06587 w_k1_IBM_s101
                                 + 1.14165 w_k1_INTC_s101
                                 + 1.28138 w_k1_KO_s101
                                 + 0.877462 w_k1_MAT_s101
                                 + 1.2086 w_k1_MSFT_s101 + 0.99435 w_k1_PG_s101
                                 >= 1
 z_k1_s1012_constraint:          var_1 + z_k1_s1012 + 0.767278 w_k1_ADM_s101
                                 + 0.636898 w_k1_AXP_s101
                                 + 0.63893 w_k1_DIS_s101
                                 + 0.789738 w_k1_FDX_s101
                                 + 0.817262 w_k1_IBM_s101
                                 + 0.958196 w_k1_INTC_s101
                                 + 0.836279 w_k1_KO_s101
                                 + 0.599814 w_k1_MAT_s101
                                 + 0.876414 w_k1_MSFT_s101
                                 + 0.795349 w_k1_PG_s101 >= 1
 z_k1_s1013_constraint:          var_1 + z_k1_s1013 + 1.19088 w_k1_ADM_s101
                                 + 1.54768 w_k1_AXP_s101
                                 + 1.42161 w_k1_DIS_s101
                                 + 1.41036 w_k1_FDX_s101
                                 + 1.16706 w_k1_IBM_s101
                                 + 1.16929 w_k1_INTC_s101
                                 + 1.44298 w_k1_KO_s101 + 1.11754 w_k1_MAT_s101
                                 + 1.45488 w_k1_MSFT_s101
                                 + 1.20076 w_k1_PG_s101 >= 1
 z_k1_s1014_constraint:          var_1 + z_k1_s1014 + 1.11378 w_k1_ADM_s101
                                 + 0.947708 w_k1_AXP_s101
                                 + 0.964079 w_k1_DIS_s101
                                 + 1.12502 w_k1_FDX_s101
                                 + 0.724856 w_k1_IBM_s101
                                 + 1.22372 w_k1_INTC_s101
                                 + 0.848679 w_k1_KO_s101
                                 + 0.582553 w_k1_MAT_s101
                                 + 1.12117 w_k1_MSFT_s101
                                 + 0.951758 w_k1_PG_s101 >= 1
 z_k1_s1015_constraint:          var_1 + z_k1_s1015 + 0.951399 w_k1_ADM_s101
                                 + 1.23645 w_k1_AXP_s101
                                 + 1.15197 w_k1_DIS_s101
                                 + 1.11908 w_k1_FDX_s101
                                 + 1.33625 w_k1_IBM_s101
                                 + 1.15834 w_k1_INTC_s101
                                 + 1.07717 w_k1_KO_s101 + 1.25795 w_k1_MAT_s101
                                 + 1.12815 w_k1_MSFT_s101 + 1.0333 w_k1_PG_s101
                                 >= 1
 z_k1_s1016_constraint:          var_1 + z_k1_s1016 + 0.974071 w_k1_ADM_s101
                                 + 1.04803 w_k1_AXP_s101
                                 + 1.17399 w_k1_DIS_s101
                                 + 1.05505 w_k1_FDX_s101
                                 + 0.875432 w_k1_IBM_s101
                                 + 1.08313 w_k1_INTC_s101
                                 + 1.17044 w_k1_KO_s101 + 1.03361 w_k1_MAT_s101
                                 + 1.11334 w_k1_MSFT_s101
                                 + 0.990442 w_k1_PG_s101 >= 1
 z_k1_s1017_constraint:          var_1 + z_k1_s1017 + 0.932043 w_k1_ADM_s101
                                 + 1.18054 w_k1_AXP_s101
                                 + 0.91039 w_k1_DIS_s101
                                 + 0.960204 w_k1_FDX_s101
                                 + 1.03763 w_k1_IBM_s101
                                 + 1.01294 w_k1_INTC_s101
                                 + 0.994247 w_k1_KO_s101
                                 + 0.994727 w_k1_MAT_s101
                                 + 0.9749 w_k1_MSFT_s101 + 0.96969 w_k1_PG_s101
                                 >= 1
 z_k1_s1020_constraint:          var_1 + z_k1_s1020 + 0.968928 w_k1_ADM_s102
                                 + 1.34814 w_k1_AXP_s102
                                 + 2.16643 w_k1_DIS_s102
                                 + 1.38141 w_k1_FDX_s102
                                 + 1.44071 w_k1_IBM_s102
                                 + 1.19966 w_k1_INTC_s102
                                 + 1.34292 w_k1_KO_s102 + 2.13929 w_k1_MAT_s102
                                 + 1.37521 w_k1_MSFT_s102
                                 + 1.22384 w_k1_PG_s102 >= 1
 z_k1_s1021_constraint:          var_1 + z_k1_s1021 + 0.598891 w_k1_ADM_s102
                                 + 0.703177 w_k1_AXP_s102
                                 + 0.692578 w_k1_DIS_s102
                                 + 0.726859 w_k1_FDX_s102
                                 + 0.99504 w_k1_IBM_s102
                                 + 0.796205 w_k1_INTC_s102
                                 + 0.720365 w_k1_KO_s102
                                 + 0.604199 w_k1_MAT_s102
                                 + 0.750536 w_k1_MSFT_s102
                                 + 0.741829 w_k1_PG_s102 >= 1
 z_k1_s1022_constraint:          var_1 + z_k1_s1022 + 0.930999 w_k1_ADM_s102
                                 + 1.29297 w_k1_AXP_s102
                                 + 1.36933 w_k1_DIS_s102
                                 + 1.07691 w_k1_FDX_s102
                                 + 1.32495 w_k1_IBM_s102
                                 + 1.10524 w_k1_INTC_s102 + 1.2379 w_k1_KO_s102
                                 + 1.21228 w_k1_MAT_s102
                                 + 1.12997 w_k1_MSFT_s102
                                 + 1.06623 w_k1_PG_s102 >= 1
 z_k1_s1023_constraint:          var_1 + z_k1_s1023 + 0.83548 w_k1_ADM_s102
                                 + 0.840712 w_k1_AXP_s102
                                 + 0.746132 w_k1_DIS_s102
                                 + 0.890264 w_k1_FDX_s102
                                 + 0.631238 w_k1_IBM_s102
                                 + 0.79858 w_k1_INTC_s102
                                 + 0.697979 w_k1_KO_s102
                                 + 0.648401 w_k1_MAT_s102
                                 + 0.756357 w_k1_MSFT_s102
                                 + 0.849805 w_k1_PG_s102 >= 1
 z_k1_s1024_constraint:          var_1 + z_k1_s1024 + 0.905217 w_k1_ADM_s102
                                 + 1.0459 w_k1_AXP_s102 + 1.02585 w_k1_DIS_s102
                                 + 0.970185 w_k1_FDX_s102
                                 + 1.07144 w_k1_IBM_s102
                                 + 1.23328 w_k1_INTC_s102
                                 + 1.08931 w_k1_KO_s102
                                 + 0.748814 w_k1_MAT_s102
                                 + 0.963803 w_k1_MSFT_s102
                                 + 1.05832 w_k1_PG_s102 >= 1
 z_k1_s1025_constraint:          var_1 + z_k1_s1025 + 1.0933 w_k1_ADM_s102
                                 + 1.58591 w_k1_AXP_s102
                                 + 1.63114 w_k1_DIS_s102
                                 + 1.31302 w_k1_FDX_s102
                                 + 1.13845 w_k1_IBM_s102
                                 + 1.33615 w_k1_INTC_s102
                                 + 1.30986 w_k1_KO_s102
                                 + 0.994424 w_k1_MAT_s102
                                 + 1.32213 w_k1_MSFT_s102
                                 + 1.16625 w_k1_PG_s102 >= 1
 z_k1_s1026_constraint:          var_1 + z_k1_s1026 + 0.931804 w_k1_ADM_s102
                                 + 1.11174 w_k1_AXP_s102
                                 + 1.04585 w_k1_DIS_s102
                                 + 1.01511 w_k1_FDX_s102
                                 + 0.870779 w_k1_IBM_s102
                                 + 1.06788 w_k1_INTC_s102
                                 + 0.986329 w_k1_KO_s102
                                 + 1.00674 w_k1_MAT_s102
                                 + 0.996418 w_k1_MSFT_s102
                                 + 0.978524 w_k1_PG_s102 >= 1
 z_k1_s1027_constraint:          var_1 + z_k1_s1027 + 0.776868 w_k1_ADM_s102
                                 + 1.40933 w_k1_AXP_s102 + 1.1415 w_k1_DIS_s102
                                 + 0.931028 w_k1_FDX_s102
                                 + 1.29352 w_k1_IBM_s102
                                 + 1.08745 w_k1_INTC_s102
                                 + 1.18008 w_k1_KO_s102 + 1.67366 w_k1_MAT_s102
                                 + 1.01267 w_k1_MSFT_s102
                                 + 1.05013 w_k1_PG_s102 >= 1
 z_k1_s1030_constraint:          var_1 + z_k1_s1030 + 1.09457 w_k1_ADM_s103
                                 + 1.43642 w_k1_AXP_s103
                                 + 1.37921 w_k1_DIS_s103
                                 + 1.44172 w_k1_FDX_s103
                                 + 0.901278 w_k1_IBM_s103
                                 + 1.43843 w_k1_INTC_s103
                                 + 1.13845 w_k1_KO_s103
                                 + 0.978914 w_k1_MAT_s103
                                 + 1.23797 w_k1_MSFT_s103
                                 + 1.05541 w_k1_PG_s103 >= 1
 z_k1_s1031_constraint:          var_1 + z_k1_s1031 + 0.83878 w_k1_ADM_s103
                                 + 0.809019 w_k1_AXP_s103
                                 + 0.634246 w_k1_DIS_s103
                                 + 0.844142 w_k1_FDX_s103
                                 + 0.667226 w_k1_IBM_s103
                                 + 0.909547 w_k1_INTC_s103
                                 + 0.690632 w_k1_KO_s103
                                 + 0.630158 w_k1_MAT_s103
                                 + 0.811915 w_k1_MSFT_s103
                                 + 0.767359 w_k1_PG_s103 >= 1
 z_k1_s1032_constraint:          var_1 + z_k1_s1032 + 1.00453 w_k1_ADM_s103
                                 + 1.03656 w_k1_AXP_s103
                                 + 1.18706 w_k1_DIS_s103
                                 + 1.08317 w_k1_FDX_s103
                                 + 1.28887 w_k1_IBM_s103
                                 + 1.03373 w_k1_INTC_s103
                                 + 1.15459 w_k1_KO_s103 + 1.56203 w_k1_MAT_s103
                                 + 1.04545 w_k1_MSFT_s103
                                 + 1.04704 w_k1_PG_s103 >= 1
 z_k1_s1033_constraint:          var_1 + z_k1_s1033 + 0.881568 w_k1_ADM_s103
                                 + 0.696603 w_k1_AXP_s103
                                 + 1.06083 w_k1_DIS_s103
                                 + 0.881279 w_k1_FDX_s103
                                 + 1.25297 w_k1_IBM_s103
                                 + 0.909579 w_k1_INTC_s103
                                 + 1.18568 w_k1_KO_s103
                                 + 0.905539 w_k1_MAT_s103
                                 + 1.09891 w_k1_MSFT_s103
                                 + 0.970151 w_k1_PG_s103 >= 1
 z_k1_s1034_constraint:          var_1 + z_k1_s1034 + 0.959333 w_k1_ADM_s103
                                 + 0.808295 w_k1_AXP_s103
                                 + 0.792821 w_k1_DIS_s103
                                 + 0.880011 w_k1_FDX_s103
                                 + 0.998411 w_k1_IBM_s103
                                 + 0.875685 w_k1_INTC_s103
                                 + 0.990311 w_k1_KO_s103
                                 + 0.826324 w_k1_MAT_s103
                                 + 0.905698 w_k1_MSFT_s103
                                 + 0.854131 w_k1_PG_s103 >= 1
 z_k1_s1035_constraint:          var_1 + z_k1_s1035 + 0.856292 w_k1_ADM_s103
                                 + 1.27467 w_k1_AXP_s103
                                 + 0.982096 w_k1_DIS_s103
                                 + 1.05771 w_k1_FDX_s103
                                 + 0.829914 w_k1_IBM_s103
                                 + 1.01017 w_k1_INTC_s103
                                 + 0.864423 w_k1_KO_s103
                                 + 1.08288 w_k1_MAT_s103
                                 + 1.07485 w_k1_MSFT_s103
                                 + 0.953749 w_k1_PG_s103 >= 1
 z_k1_s1036_constraint:          var_1 + z_k1_s1036 + 1.17328 w_k1_ADM_s103
                                 + 1.27382 w_k1_AXP_s103
                                 + 1.53054 w_k1_DIS_s103
                                 + 1.22083 w_k1_FDX_s103
                                 + 1.83301 w_k1_IBM_s103
                                 + 1.29211 w_k1_INTC_s103
                                 + 1.42113 w_k1_KO_s103 + 1.6902 w_k1_MAT_s103
                                 + 1.21775 w_k1_MSFT_s103
                                 + 1.10939 w_k1_PG_s103 >= 1
 z_k1_s1037_constraint:          var_1 + z_k1_s1037 + 1.27654 w_k1_ADM_s103
                                 + 0.964687 w_k1_AXP_s103
                                 + 1.16468 w_k1_DIS_s103
                                 + 1.20153 w_k1_FDX_s103
                                 + 0.880335 w_k1_IBM_s103
                                 + 1.13455 w_k1_INTC_s103
                                 + 1.04041 w_k1_KO_s103
                                 + 0.867871 w_k1_MAT_s103
                                 + 1.11382 w_k1_MSFT_s103
                                 + 0.989386 w_k1_PG_s103 >= 1
 z_k1_s1040_constraint:          var_1 + z_k1_s1040 + 0.729848 w_k1_ADM_s104
                                 + 0.842666 w_k1_AXP_s104
                                 + 0.814681 w_k1_DIS_s104
                                 + 0.79331 w_k1_FDX_s104
                                 + 0.874344 w_k1_IBM_s104
                                 + 0.853636 w_k1_INTC_s104
                                 + 0.877824 w_k1_KO_s104
                                 + 0.86867 w_k1_MAT_s104
                                 + 0.854719 w_k1_MSFT_s104
                                 + 0.901012 w_k1_PG_s104 >= 1
 z_k1_s1041_constraint:          var_1 + z_k1_s1041 + 0.981978 w_k1_ADM_s104
                                 + 1.03917 w_k1_AXP_s104
                                 + 1.37202 w_k1_DIS_s104
                                 + 0.978357 w_k1_FDX_s104
                                 + 1.28618 w_k1_IBM_s104
                                 + 1.17669 w_k1_INTC_s104
                                 + 1.54743 w_k1_KO_s104 + 1.08954 w_k1_MAT_s104
                                 + 1.28458 w_k1_MSFT_s104
                                 + 1.10559 w_k1_PG_s104 >= 1
 z_k1_s1042_constraint:          var_1 + z_k1_s1042 + 1.13302 w_k1_ADM_s104
                                 + 1.10313 w_k1_AXP_s104
                                 + 1.11996 w_k1_DIS_s104
                                 + 1.03203 w_k1_FDX_s104
                                 + 0.870575 w_k1_IBM_s104
                                 + 0.880144 w_k1_INTC_s104
                                 + 1.02858 w_k1_KO_s104
                                 + 0.698838 w_k1_MAT_s104
                                 + 1.10553 w_k1_MSFT_s104
                                 + 0.927382 w_k1_PG_s104 >= 1
 z_k1_s1043_constraint:          var_1 + z_k1_s1043 + 0.957766 w_k1_ADM_s104
                                 + 0.742973 w_k1_AXP_s104
                                 + 0.619691 w_k1_DIS_s104
                                 + 0.81063 w_k1_FDX_s104
                                 + 0.536492 w_k1_IBM_s104
                                 + 0.786929 w_k1_INTC_s104
                                 + 0.603124 w_k1_KO_s104
                                 + 0.312407 w_k1_MAT_s104
                                 + 0.842934 w_k1_MSFT_s104
                                 + 0.800985 w_k1_PG_s104 >= 1
 z_k1_s1044_constraint:          var_1 + z_k1_s1044 + 1.03649 w_k1_ADM_s104
                                 + 1.00943 w_k1_AXP_s104
                                 + 0.810651 w_k1_DIS_s104
                                 + 0.873238 w_k1_FDX_s104
                                 + 0.793374 w_k1_IBM_s104
                                 + 0.793149 w_k1_INTC_s104
                                 + 0.881193 w_k1_KO_s104
                                 + 0.666826 w_k1_MAT_s104
                                 + 0.891179 w_k1_MSFT_s104
                                 + 0.901452 w_k1_PG_s104 >= 1
 z_k1_s1045_constraint:          var_1 + z_k1_s1045 + 0.989129 w_k1_ADM_s104
                                 + 1.15788 w_k1_AXP_s104
                                 + 1.20501 w_k1_DIS_s104
                                 + 1.00662 w_k1_FDX_s104
                                 + 1.47358 w_k1_IBM_s104
                                 + 0.894853 w_k1_INTC_s104
                                 + 1.37028 w_k1_KO_s104 + 1.37526 w_k1_MAT_s104
                                 + 1.01508 w_k1_MSFT_s104
                                 + 1.16198 w_k1_PG_s104 >= 1
 z_k1_s1046_constraint:          var_1 + z_k1_s1046 + 0.72269 w_k1_ADM_s104
                                 + 0.659157 w_k1_AXP_s104 + 0.543 w_k1_DIS_s104
                                 + 0.67821 w_k1_FDX_s104
                                 + 0.715895 w_k1_IBM_s104
                                 + 0.720547 w_k1_INTC_s104
                                 + 0.647079 w_k1_KO_s104
                                 + 0.57997 w_k1_MAT_s104
                                 + 0.666686 w_k1_MSFT_s104
                                 + 0.743347 w_k1_PG_s104 >= 1
 z_k1_s1047_constraint:          var_1 + z_k1_s1047 + 1.18715 w_k1_ADM_s104
                                 + 1.47475 w_k1_AXP_s104
                                 + 1.90047 w_k1_DIS_s104
                                 + 1.40189 w_k1_FDX_s104
                                 + 1.16758 w_k1_IBM_s104
                                 + 1.28768 w_k1_INTC_s104 + 1.4139 w_k1_KO_s104
                                 + 1.24184 w_k1_MAT_s104
                                 + 1.42227 w_k1_MSFT_s104
                                 + 1.12825 w_k1_PG_s104 >= 1
 z_k1_s1050_constraint:          var_1 + z_k1_s1050 + 0.835694 w_k1_ADM_s105
                                 + 0.927694 w_k1_AXP_s105
                                 + 0.89824 w_k1_DIS_s105
                                 + 0.816834 w_k1_FDX_s105
                                 + 1.23829 w_k1_IBM_s105
                                 + 0.991599 w_k1_INTC_s105
                                 + 0.968058 w_k1_KO_s105
                                 + 1.09459 w_k1_MAT_s105
                                 + 0.764134 w_k1_MSFT_s105
                                 + 0.989524 w_k1_PG_s105 >= 1
 z_k1_s1051_constraint:          var_1 + z_k1_s1051 + 1.06144 w_k1_ADM_s105
                                 + 1.05349 w_k1_AXP_s105
                                 + 1.22818 w_k1_DIS_s105
                                 + 1.12821 w_k1_FDX_s105
                                 + 0.740105 w_k1_IBM_s105
                                 + 1.05035 w_k1_INTC_s105
                                 + 1.09983 w_k1_KO_s105
                                 + 0.687462 w_k1_MAT_s105
                                 + 1.11949 w_k1_MSFT_s105
                                 + 1.05962 w_k1_PG_s105 >= 1
 z_k1_s1052_constraint:          var_1 + z_k1_s1052 + 0.846785 w_k1_ADM_s105
                                 + 0.68886 w_k1_AXP_s105
                                 + 0.716321 w_k1_DIS_s105
                                 + 0.760402 w_k1_FDX_s105
                                 + 0.806719 w_k1_IBM_s105
                                 + 0.91166 w_k1_INTC_s105
                                 + 0.853965 w_k1_KO_s105
                                 + 0.884417 w_k1_MAT_s105
                                 + 0.776984 w_k1_MSFT_s105
                                 + 0.96938 w_k1_PG_s105 >= 1
 z_k1_s1053_constraint:          var_1 + z_k1_s1053 + 0.983339 w_k1_ADM_s105
                                 + 1.23529 w_k1_AXP_s105
                                 + 0.985796 w_k1_DIS_s105
                                 + 0.982677 w_k1_FDX_s105
                                 + 1.01464 w_k1_IBM_s105
                                 + 1.05738 w_k1_INTC_s105
                                 + 0.994208 w_k1_KO_s105
                                 + 1.30155 w_k1_MAT_s105
                                 + 0.940341 w_k1_MSFT_s105
                                 + 1.1588 w_k1_PG_s105 >= 1
 z_k1_s1054_constraint:          var_1 + z_k1_s1054 + 0.771697 w_k1_ADM_s105
                                 + 0.755155 w_k1_AXP_s105
                                 + 0.869526 w_k1_DIS_s105
                                 + 0.884838 w_k1_FDX_s105
                                 + 0.955222 w_k1_IBM_s105
                                 + 1.05216 w_k1_INTC_s105
                                 + 0.961565 w_k1_KO_s105
                                 + 0.846078 w_k1_MAT_s105
                                 + 0.908747 w_k1_MSFT_s105
                                 + 1.08626 w_k1_PG_s105 >= 1
 z_k1_s1055_constraint:          var_1 + z_k1_s1055 + 0.711895 w_k1_ADM_s105
                                 + 0.726518 w_k1_AXP_s105
                                 + 0.81242 w_k1_DIS_s105
                                 + 0.835842 w_k1_FDX_s105
                                 + 0.570415 w_k1_IBM_s105
                                 + 0.92671 w_k1_INTC_s105
                                 + 0.745096 w_k1_KO_s105
                                 + 0.663783 w_k1_MAT_s105
                                 + 0.80205 w_k1_MSFT_s105
                                 + 0.935422 w_k1_PG_s105 >= 1
 z_k1_s1056_constraint:          var_1 + z_k1_s1056 + 1.20031 w_k1_ADM_s105
                                 + 1.34294 w_k1_AXP_s105 + 1.5998 w_k1_DIS_s105
                                 + 1.37481 w_k1_FDX_s105 + 1.0806 w_k1_IBM_s105
                                 + 1.10798 w_k1_INTC_s105
                                 + 1.29827 w_k1_KO_s105 + 1.84887 w_k1_MAT_s105
                                 + 1.12261 w_k1_MSFT_s105
                                 + 1.32124 w_k1_PG_s105 >= 1
 z_k1_s1057_constraint:          var_1 + z_k1_s1057 + 0.899564 w_k1_ADM_s105
                                 + 1.04222 w_k1_AXP_s105 + 1.5351 w_k1_DIS_s105
                                 + 1.05478 w_k1_FDX_s105
                                 + 1.54445 w_k1_IBM_s105
                                 + 1.1012 w_k1_INTC_s105 + 1.49189 w_k1_KO_s105
                                 + 1.72927 w_k1_MAT_s105
                                 + 1.14134 w_k1_MSFT_s105
                                 + 1.28571 w_k1_PG_s105 >= 1
 z_k1_s1060_constraint:          var_1 + z_k1_s1060 + 0.696697 w_k1_ADM_s106
                                 + 0.820602 w_k1_AXP_s106
                                 + 0.794159 w_k1_DIS_s106
                                 + 0.804366 w_k1_FDX_s106
                                 + 1.25518 w_k1_IBM_s106
                                 + 1.07772 w_k1_INTC_s106
                                 + 1.02591 w_k1_KO_s106 + 1.27797 w_k1_MAT_s106
                                 + 0.938464 w_k1_MSFT_s106
                                 + 1.01669 w_k1_PG_s106 >= 1
 z_k1_s1061_constraint:          var_1 + z_k1_s1061 + 0.69647 w_k1_ADM_s106
                                 + 0.619163 w_k1_AXP_s106
                                 + 0.627342 w_k1_DIS_s106
                                 + 0.765142 w_k1_FDX_s106
                                 + 0.730347 w_k1_IBM_s106
                                 + 0.919718 w_k1_INTC_s106
                                 + 0.713054 w_k1_KO_s106
                                 + 0.688085 w_k1_MAT_s106
                                 + 0.812683 w_k1_MSFT_s106
                                 + 0.755979 w_k1_PG_s106 >= 1
 z_k1_s1062_constraint:          var_1 + z_k1_s1062 + 1.35835 w_k1_ADM_s106
                                 + 1.39632 w_k1_AXP_s106
                                 + 1.17734 w_k1_DIS_s106
                                 + 1.22034 w_k1_FDX_s106
                                 + 1.06977 w_k1_IBM_s106
                                 + 1.2383 w_k1_INTC_s106 + 1.06371 w_k1_KO_s106
                                 + 1.29582 w_k1_MAT_s106
                                 + 1.18535 w_k1_MSFT_s106
                                 + 1.19433 w_k1_PG_s106 >= 1
 z_k1_s1063_constraint:          var_1 + z_k1_s1063 + 1.0517 w_k1_ADM_s106
                                 + 0.829091 w_k1_AXP_s106
                                 + 1.11468 w_k1_DIS_s106
                                 + 1.08477 w_k1_FDX_s106
                                 + 0.797977 w_k1_IBM_s106
                                 + 1.04367 w_k1_INTC_s106
                                 + 0.936291 w_k1_KO_s106
                                 + 1.08156 w_k1_MAT_s106
                                 + 1.22612 w_k1_MSFT_s106
                                 + 1.01615 w_k1_PG_s106 >= 1
 z_k1_s1064_constraint:          var_1 + z_k1_s1064 + 1.29042 w_k1_ADM_s106
                                 + 1.49199 w_k1_AXP_s106
                                 + 1.54931 w_k1_DIS_s106
                                 + 1.37511 w_k1_FDX_s106
                                 + 1.05861 w_k1_IBM_s106
                                 + 1.10686 w_k1_INTC_s106
                                 + 1.20209 w_k1_KO_s106 + 1.42153 w_k1_MAT_s106
                                 + 1.2062 w_k1_MSFT_s106 + 1.09648 w_k1_PG_s106
                                 >= 1
 z_k1_s1065_constraint:          var_1 + z_k1_s1065 + 1.05124 w_k1_ADM_s106
                                 + 0.95983 w_k1_AXP_s106
                                 + 0.873089 w_k1_DIS_s106
                                 + 0.979884 w_k1_FDX_s106
                                 + 1.01377 w_k1_IBM_s106
                                 + 1.04755 w_k1_INTC_s106
                                 + 0.89934 w_k1_KO_s106
                                 + 0.809505 w_k1_MAT_s106
                                 + 1.04184 w_k1_MSFT_s106
                                 + 0.980782 w_k1_PG_s106 >= 1
 z_k1_s1066_constraint:          var_1 + z_k1_s1066 + 1.11001 w_k1_ADM_s106
                                 + 1.08701 w_k1_AXP_s106
                                 + 1.42144 w_k1_DIS_s106
                                 + 1.08013 w_k1_FDX_s106
                                 + 1.61329 w_k1_IBM_s106
                                 + 1.26525 w_k1_INTC_s106
                                 + 1.63057 w_k1_KO_s106 + 1.63708 w_k1_MAT_s106
                                 + 1.3303 w_k1_MSFT_s106 + 1.19524 w_k1_PG_s106
                                 >= 1
 z_k1_s1067_constraint:          var_1 + z_k1_s1067 + 1.11914 w_k1_ADM_s106
                                 + 0.904826 w_k1_AXP_s106
                                 + 0.749554 w_k1_DIS_s106
                                 + 0.941661 w_k1_FDX_s106
                                 + 0.634867 w_k1_IBM_s106
                                 + 0.899949 w_k1_INTC_s106
                                 + 0.688661 w_k1_KO_s106
                                 + 0.89331 w_k1_MAT_s106
                                 + 0.96534 w_k1_MSFT_s106
                                 + 0.862751 w_k1_PG_s106 >= 1
 z_k1_s1070_constraint:          var_1 + z_k1_s1070 + 1.08679 w_k1_ADM_s107
                                 + 1.6396 w_k1_AXP_s107 + 1.11464 w_k1_DIS_s107
                                 + 1.2592 w_k1_FDX_s107 + 1.23685 w_k1_IBM_s107
                                 + 1.16519 w_k1_INTC_s107
                                 + 1.03325 w_k1_KO_s107 + 1.50033 w_k1_MAT_s107
                                 + 1.05185 w_k1_MSFT_s107
                                 + 1.20281 w_k1_PG_s107 >= 1
 z_k1_s1071_constraint:          var_1 + z_k1_s1071 + 0.971757 w_k1_ADM_s107
                                 + 1.24553 w_k1_AXP_s107 + 1.639 w_k1_DIS_s107
                                 + 1.45476 w_k1_FDX_s107
                                 + 1.36187 w_k1_IBM_s107
                                 + 1.16774 w_k1_INTC_s107 + 1.5177 w_k1_KO_s107
                                 + 1.46938 w_k1_MAT_s107
                                 + 1.30341 w_k1_MSFT_s107
                                 + 1.31636 w_k1_PG_s107 >= 1
 z_k1_s1072_constraint:          var_1 + z_k1_s1072 + 0.717304 w_k1_ADM_s107
                                 + 0.72613 w_k1_AXP_s107
                                 + 0.570215 w_k1_DIS_s107
                                 + 0.770302 w_k1_FDX_s107
                                 + 0.783073 w_k1_IBM_s107
                                 + 0.809115 w_k1_INTC_s107
                                 + 0.714523 w_k1_KO_s107
                                 + 0.53094 w_k1_MAT_s107
                                 + 0.792238 w_k1_MSFT_s107
                                 + 0.829779 w_k1_PG_s107 >= 1
 z_k1_s1073_constraint:          var_1 + z_k1_s1073 + 0.742386 w_k1_ADM_s107
                                 + 0.836684 w_k1_AXP_s107
                                 + 0.946229 w_k1_DIS_s107
                                 + 0.985882 w_k1_FDX_s107
                                 + 0.769574 w_k1_IBM_s107
                                 + 0.794502 w_k1_INTC_s107
                                 + 0.813984 w_k1_KO_s107
                                 + 0.833835 w_k1_MAT_s107
                                 + 0.937611 w_k1_MSFT_s107
                                 + 0.91863 w_k1_PG_s107 >= 1
 z_k1_s1074_constraint:          var_1 + z_k1_s1074 + 1.01619 w_k1_ADM_s107
                                 + 0.915923 w_k1_AXP_s107
                                 + 1.17001 w_k1_DIS_s107
                                 + 1.00079 w_k1_FDX_s107 + 1.2192 w_k1_IBM_s107
                                 + 0.9716 w_k1_INTC_s107 + 1.32344 w_k1_KO_s107
                                 + 1.3697 w_k1_MAT_s107
                                 + 1.18565 w_k1_MSFT_s107
                                 + 1.12044 w_k1_PG_s107 >= 1
 z_k1_s1075_constraint:          var_1 + z_k1_s1075 + 1.20776 w_k1_ADM_s107
                                 + 1.18382 w_k1_AXP_s107
                                 + 1.56776 w_k1_DIS_s107
                                 + 1.23194 w_k1_FDX_s107
                                 + 1.83912 w_k1_IBM_s107
                                 + 1.19435 w_k1_INTC_s107
                                 + 1.61641 w_k1_KO_s107 + 1.2045 w_k1_MAT_s107
                                 + 1.13948 w_k1_MSFT_s107
                                 + 1.26902 w_k1_PG_s107 >= 1
 z_k1_s1076_constraint:          var_1 + z_k1_s1076 + 1.14418 w_k1_ADM_s107
                                 + 1.11345 w_k1_AXP_s107
                                 + 1.35422 w_k1_DIS_s107
                                 + 1.24438 w_k1_FDX_s107
                                 + 0.989991 w_k1_IBM_s107
                                 + 1.23904 w_k1_INTC_s107
                                 + 1.20648 w_k1_KO_s107
                                 + 0.931275 w_k1_MAT_s107
                                 + 1.35225 w_k1_MSFT_s107
                                 + 1.14972 w_k1_PG_s107 >= 1
 z_k1_s1077_constraint:          var_1 + z_k1_s1077 + 1.05767 w_k1_ADM_s107
                                 + 0.926046 w_k1_AXP_s107
                                 + 0.956353 w_k1_DIS_s107
                                 + 0.914575 w_k1_FDX_s107
                                 + 1.24749 w_k1_IBM_s107
                                 + 0.930221 w_k1_INTC_s107
                                 + 1.19218 w_k1_KO_s107 + 0.96751 w_k1_MAT_s107
                                 + 0.981944 w_k1_MSFT_s107
                                 + 1.08612 w_k1_PG_s107 >= 1
 z_k1_s1100_constraint:          var_1 + z_k1_s1100 + 1.37271 w_k1_ADM_s110
                                 + 1.46132 w_k1_AXP_s110
                                 + 1.42806 w_k1_DIS_s110
                                 + 1.31866 w_k1_FDX_s110
                                 + 1.00662 w_k1_IBM_s110
                                 + 1.40887 w_k1_INTC_s110
                                 + 1.18231 w_k1_KO_s110 + 1.13333 w_k1_MAT_s110
                                 + 1.41698 w_k1_MSFT_s110
                                 + 1.21073 w_k1_PG_s110 >= 1
 z_k1_s1101_constraint:          var_1 + z_k1_s1101 + 0.976005 w_k1_ADM_s110
                                 + 1.14437 w_k1_AXP_s110
                                 + 1.06803 w_k1_DIS_s110
                                 + 1.00003 w_k1_FDX_s110
                                 + 1.06971 w_k1_IBM_s110
                                 + 1.0724 w_k1_INTC_s110
                                 + 0.971932 w_k1_KO_s110
                                 + 1.15126 w_k1_MAT_s110
                                 + 1.03479 w_k1_MSFT_s110
                                 + 0.958761 w_k1_PG_s110 >= 1
 z_k1_s1102_constraint:          var_1 + z_k1_s1102 + 1.31555 w_k1_ADM_s110
                                 + 1.64644 w_k1_AXP_s110
                                 + 1.55901 w_k1_DIS_s110
                                 + 1.29016 w_k1_FDX_s110
                                 + 1.24971 w_k1_IBM_s110
                                 + 1.33081 w_k1_INTC_s110
                                 + 1.20259 w_k1_KO_s110 + 1.59374 w_k1_MAT_s110
                                 + 1.27853 w_k1_MSFT_s110
                                 + 1.32403 w_k1_PG_s110 >= 1
 z_k1_s1103_constraint:          var_1 + z_k1_s1103 + 0.799314 w_k1_ADM_s110
                                 + 0.662269 w_k1_AXP_s110
                                 + 0.602608 w_k1_DIS_s110
                                 + 0.683249 w_k1_FDX_s110
                                 + 0.639983 w_k1_IBM_s110
                                 + 0.777589 w_k1_INTC_s110
                                 + 0.715079 w_k1_KO_s110
                                 + 0.862454 w_k1_MAT_s110
                                 + 0.774883 w_k1_MSFT_s110
                                 + 0.767501 w_k1_PG_s110 >= 1
 z_k1_s1104_constraint:          var_1 + z_k1_s1104 + 0.88808 w_k1_ADM_s110
                                 + 0.852855 w_k1_AXP_s110
                                 + 0.802256 w_k1_DIS_s110
                                 + 0.781979 w_k1_FDX_s110
                                 + 0.941162 w_k1_IBM_s110
                                 + 0.791156 w_k1_INTC_s110
                                 + 0.766024 w_k1_KO_s110
                                 + 1.11523 w_k1_MAT_s110
                                 + 0.833353 w_k1_MSFT_s110
                                 + 0.851589 w_k1_PG_s110 >= 1
 z_k1_s1105_constraint:          var_1 + z_k1_s1105 + 1.1201 w_k1_ADM_s110
                                 + 0.931203 w_k1_AXP_s110
                                 + 1.07035 w_k1_DIS_s110
                                 + 1.05166 w_k1_FDX_s110
                                 + 0.80773 w_k1_IBM_s110
                                 + 0.949163 w_k1_INTC_s110
                                 + 0.78676 w_k1_KO_s110
                                 + 0.839559 w_k1_MAT_s110
                                 + 1.05691 w_k1_MSFT_s110
                                 + 0.913215 w_k1_PG_s110 >= 1
 z_k1_s1106_constraint:          var_1 + z_k1_s1106 + 1.08341 w_k1_ADM_s110
                                 + 1.08215 w_k1_AXP_s110
                                 + 1.31067 w_k1_DIS_s110
                                 + 1.03835 w_k1_FDX_s110
                                 + 1.48777 w_k1_IBM_s110
                                 + 0.962715 w_k1_INTC_s110
                                 + 1.19234 w_k1_KO_s110 + 1.69632 w_k1_MAT_s110
                                 + 1.15053 w_k1_MSFT_s110
                                 + 1.03025 w_k1_PG_s110 >= 1
 z_k1_s1107_constraint:          var_1 + z_k1_s1107 + 0.738157 w_k1_ADM_s110
                                 + 0.707964 w_k1_AXP_s110
                                 + 0.692648 w_k1_DIS_s110
                                 + 0.715525 w_k1_FDX_s110
                                 + 0.912385 w_k1_IBM_s110
                                 + 0.880672 w_k1_INTC_s110
                                 + 0.808682 w_k1_KO_s110
                                 + 0.783581 w_k1_MAT_s110
                                 + 0.868749 w_k1_MSFT_s110
                                 + 0.76427 w_k1_PG_s110 >= 1
 z_k1_s1110_constraint:          var_1 + z_k1_s1110 + 1.03283 w_k1_ADM_s111
                                 + 1.31707 w_k1_AXP_s111 + 1.5772 w_k1_DIS_s111
                                 + 1.33037 w_k1_FDX_s111
                                 + 1.06384 w_k1_IBM_s111
                                 + 1.12406 w_k1_INTC_s111
                                 + 1.34407 w_k1_KO_s111
                                 + 0.816865 w_k1_MAT_s111
                                 + 1.20798 w_k1_MSFT_s111
                                 + 1.21093 w_k1_PG_s111 >= 1
 z_k1_s1111_constraint:          var_1 + z_k1_s1111 + 1.03899 w_k1_ADM_s111
                                 + 0.99881 w_k1_AXP_s111
                                 + 0.735277 w_k1_DIS_s111
                                 + 0.946112 w_k1_FDX_s111
                                 + 0.720459 w_k1_IBM_s111
                                 + 0.726175 w_k1_INTC_s111
                                 + 0.715571 w_k1_KO_s111
                                 + 0.689027 w_k1_MAT_s111
                                 + 0.800496 w_k1_MSFT_s111
                                 + 0.89659 w_k1_PG_s111 >= 1
 z_k1_s1112_constraint:          var_1 + z_k1_s1112 + 0.577532 w_k1_ADM_s111
                                 + 0.659331 w_k1_AXP_s111
                                 + 0.594401 w_k1_DIS_s111
                                 + 0.740069 w_k1_FDX_s111
                                 + 0.785602 w_k1_IBM_s111
                                 + 0.710889 w_k1_INTC_s111
                                 + 0.735126 w_k1_KO_s111
                                 + 0.867286 w_k1_MAT_s111
                                 + 0.655398 w_k1_MSFT_s111
                                 + 0.856425 w_k1_PG_s111 >= 1
 z_k1_s1113_constraint:          var_1 + z_k1_s1113 + 0.958636 w_k1_ADM_s111
                                 + 0.942115 w_k1_AXP_s111
                                 + 0.986296 w_k1_DIS_s111 + 1.087 w_k1_FDX_s111
                                 + 0.662492 w_k1_IBM_s111
                                 + 1.05487 w_k1_INTC_s111
                                 + 0.974081 w_k1_KO_s111
                                 + 0.688258 w_k1_MAT_s111
                                 + 0.966749 w_k1_MSFT_s111
                                 + 1.04514 w_k1_PG_s111 >= 1
 z_k1_s1114_constraint:          var_1 + z_k1_s1114 + 0.906592 w_k1_ADM_s111
                                 + 0.79323 w_k1_AXP_s111
                                 + 0.957447 w_k1_DIS_s111
                                 + 0.94903 w_k1_FDX_s111
                                 + 0.915201 w_k1_IBM_s111
                                 + 0.824321 w_k1_INTC_s111
                                 + 0.950277 w_k1_KO_s111
                                 + 0.814066 w_k1_MAT_s111
                                 + 0.890744 w_k1_MSFT_s111
                                 + 1.00636 w_k1_PG_s111 >= 1
 z_k1_s1115_constraint:          var_1 + z_k1_s1115 + 0.79893 w_k1_ADM_s111
                                 + 1.40096 w_k1_AXP_s111
                                 + 1.31224 w_k1_DIS_s111
                                 + 1.14926 w_k1_FDX_s111
                                 + 1.53693 w_k1_IBM_s111
                                 + 1.06169 w_k1_INTC_s111
                                 + 1.35132 w_k1_KO_s111 + 1.54402 w_k1_MAT_s111
                                 + 1.0469 w_k1_MSFT_s111 + 1.0282 w_k1_PG_s111
                                 >= 1
 z_k1_s1116_constraint:          var_1 + z_k1_s1116 + 0.751436 w_k1_ADM_s111
                                 + 1.01104 w_k1_AXP_s111
                                 + 0.913848 w_k1_DIS_s111
                                 + 1.00252 w_k1_FDX_s111
                                 + 1.14263 w_k1_IBM_s111
                                 + 0.880403 w_k1_INTC_s111
                                 + 1.00141 w_k1_KO_s111 + 1.00787 w_k1_MAT_s111
                                 + 0.867198 w_k1_MSFT_s111
                                 + 1.07672 w_k1_PG_s111 >= 1
 z_k1_s1117_constraint:          var_1 + z_k1_s1117 + 0.895174 w_k1_ADM_s111
                                 + 1.27694 w_k1_AXP_s111
                                 + 1.15583 w_k1_DIS_s111
                                 + 1.19327 w_k1_FDX_s111
                                 + 0.839197 w_k1_IBM_s111
                                 + 1.16527 w_k1_INTC_s111 + 1.0057 w_k1_KO_s111
                                 + 0.816048 w_k1_MAT_s111
                                 + 0.985442 w_k1_MSFT_s111
                                 + 0.913119 w_k1_PG_s111 >= 1
 z_k1_s1120_constraint:          var_1 + z_k1_s1120 + 1.00452 w_k1_ADM_s112
                                 + 1.25051 w_k1_AXP_s112
                                 + 0.994759 w_k1_DIS_s112
                                 + 0.96858 w_k1_FDX_s112
                                 + 0.835112 w_k1_IBM_s112
                                 + 1.0564 w_k1_INTC_s112 + 1.06469 w_k1_KO_s112
                                 + 1.4258 w_k1_MAT_s112
                                 + 0.991579 w_k1_MSFT_s112
                                 + 1.00154 w_k1_PG_s112 >= 1
 z_k1_s1121_constraint:          var_1 + z_k1_s1121 + 0.814571 w_k1_ADM_s112
                                 + 0.910602 w_k1_AXP_s112
                                 + 0.779487 w_k1_DIS_s112
                                 + 0.894459 w_k1_FDX_s112
                                 + 0.772829 w_k1_IBM_s112
                                 + 1.07214 w_k1_INTC_s112
                                 + 0.877282 w_k1_KO_s112
                                 + 0.930447 w_k1_MAT_s112
                                 + 0.984827 w_k1_MSFT_s112
                                 + 0.947322 w_k1_PG_s112 >= 1
 z_k1_s1122_constraint:          var_1 + z_k1_s1122 + 0.92087 w_k1_ADM_s112
                                 + 1.01823 w_k1_AXP_s112
                                 + 1.19216 w_k1_DIS_s112
                                 + 1.02973 w_k1_FDX_s112
                                 + 1.02619 w_k1_IBM_s112
                                 + 1.07149 w_k1_INTC_s112
                                 + 1.20825 w_k1_KO_s112 + 1.2467 w_k1_MAT_s112
                                 + 1.08488 w_k1_MSFT_s112
                                 + 0.990143 w_k1_PG_s112 >= 1
 z_k1_s1123_constraint:          var_1 + z_k1_s1123 + 0.778947 w_k1_ADM_s112
                                 + 0.804863 w_k1_AXP_s112
                                 + 0.717099 w_k1_DIS_s112
                                 + 0.876077 w_k1_FDX_s112
                                 + 0.590998 w_k1_IBM_s112
                                 + 0.995743 w_k1_INTC_s112
                                 + 0.74212 w_k1_KO_s112
                                 + 0.773472 w_k1_MAT_s112
                                 + 0.890237 w_k1_MSFT_s112
                                 + 0.810647 w_k1_PG_s112 >= 1
 z_k1_s1124_constraint:          var_1 + z_k1_s1124 + 0.698399 w_k1_ADM_s112
                                 + 0.851863 w_k1_AXP_s112
                                 + 0.683199 w_k1_DIS_s112
                                 + 0.668573 w_k1_FDX_s112
                                 + 0.895049 w_k1_IBM_s112
                                 + 0.906608 w_k1_INTC_s112
                                 + 0.849599 w_k1_KO_s112
                                 + 0.985444 w_k1_MAT_s112
                                 + 0.797391 w_k1_MSFT_s112
                                 + 0.870973 w_k1_PG_s112 >= 1
 z_k1_s1125_constraint:          var_1 + z_k1_s1125 + 1.19014 w_k1_ADM_s112
                                 + 1.43831 w_k1_AXP_s112
                                 + 1.70292 w_k1_DIS_s112
                                 + 1.33419 w_k1_FDX_s112
                                 + 0.884076 w_k1_IBM_s112
                                 + 1.54828 w_k1_INTC_s112
                                 + 1.26436 w_k1_KO_s112 + 0.95554 w_k1_MAT_s112
                                 + 1.46999 w_k1_MSFT_s112
                                 + 1.08305 w_k1_PG_s112 >= 1
 z_k1_s1126_constraint:          var_1 + z_k1_s1126 + 1.1991 w_k1_ADM_s112
                                 + 1.36055 w_k1_AXP_s112
                                 + 1.72061 w_k1_DIS_s112
                                 + 1.16217 w_k1_FDX_s112
                                 + 1.64964 w_k1_IBM_s112
                                 + 1.25926 w_k1_INTC_s112
                                 + 1.55785 w_k1_KO_s112 + 1.83476 w_k1_MAT_s112
                                 + 1.43586 w_k1_MSFT_s112
                                 + 1.11583 w_k1_PG_s112 >= 1
 z_k1_s1127_constraint:          var_1 + z_k1_s1127 + 0.975569 w_k1_ADM_s112
                                 + 1.27831 w_k1_AXP_s112
                                 + 1.14735 w_k1_DIS_s112
                                 + 0.97244 w_k1_FDX_s112
                                 + 1.33464 w_k1_IBM_s112
                                 + 1.17851 w_k1_INTC_s112
                                 + 1.36129 w_k1_KO_s112 + 1.73208 w_k1_MAT_s112
                                 + 0.964067 w_k1_MSFT_s112
                                 + 0.971375 w_k1_PG_s112 >= 1
 z_k1_s1130_constraint:          var_1 + z_k1_s1130 + 0.604079 w_k1_ADM_s113
                                 + 0.683118 w_k1_AXP_s113
                                 + 0.719192 w_k1_DIS_s113
                                 + 0.792587 w_k1_FDX_s113
                                 + 0.66638 w_k1_IBM_s113
                                 + 0.908892 w_k1_INTC_s113
                                 + 0.841469 w_k1_KO_s113
                                 + 0.826481 w_k1_MAT_s113
                                 + 0.909652 w_k1_MSFT_s113
                                 + 0.794905 w_k1_PG_s113 >= 1
 z_k1_s1131_constraint:          var_1 + z_k1_s1131 + 0.808844 w_k1_ADM_s113
                                 + 0.794236 w_k1_AXP_s113
                                 + 0.937844 w_k1_DIS_s113
                                 + 0.780321 w_k1_FDX_s113
                                 + 1.20056 w_k1_IBM_s113
                                 + 0.940643 w_k1_INTC_s113
                                 + 1.26839 w_k1_KO_s113 + 1.23995 w_k1_MAT_s113
                                 + 0.967505 w_k1_MSFT_s113
                                 + 0.895605 w_k1_PG_s113 >= 1
 z_k1_s1132_constraint:          var_1 + z_k1_s1132 + 1.01541 w_k1_ADM_s113
                                 + 1.08052 w_k1_AXP_s113
                                 + 0.799061 w_k1_DIS_s113
                                 + 1.01302 w_k1_FDX_s113
                                 + 0.773612 w_k1_IBM_s113
                                 + 0.959156 w_k1_INTC_s113
                                 + 0.829355 w_k1_KO_s113
                                 + 0.880613 w_k1_MAT_s113
                                 + 0.931352 w_k1_MSFT_s113
                                 + 0.823424 w_k1_PG_s113 >= 1
 z_k1_s1133_constraint:          var_1 + z_k1_s1133 + 0.568482 w_k1_ADM_s113
                                 + 0.829743 w_k1_AXP_s113
                                 + 0.660682 w_k1_DIS_s113
                                 + 0.72106 w_k1_FDX_s113
                                 + 1.01055 w_k1_IBM_s113
                                 + 0.945081 w_k1_INTC_s113
                                 + 0.867887 w_k1_KO_s113
                                 + 0.904466 w_k1_MAT_s113
                                 + 0.745158 w_k1_MSFT_s113
                                 + 0.715671 w_k1_PG_s113 >= 1
 z_k1_s1134_constraint:          var_1 + z_k1_s1134 + 0.854568 w_k1_ADM_s113
                                 + 1.10548 w_k1_AXP_s113
                                 + 0.960617 w_k1_DIS_s113
                                 + 0.959659 w_k1_FDX_s113
                                 + 1.01121 w_k1_IBM_s113
                                 + 0.985778 w_k1_INTC_s113
                                 + 1.07791 w_k1_KO_s113 + 0.84878 w_k1_MAT_s113
                                 + 0.978153 w_k1_MSFT_s113
                                 + 0.976768 w_k1_PG_s113 >= 1
 z_k1_s1135_constraint:          var_1 + z_k1_s1135 + 0.94246 w_k1_ADM_s113
                                 + 1.08293 w_k1_AXP_s113
                                 + 1.33961 w_k1_DIS_s113
                                 + 1.15536 w_k1_FDX_s113
                                 + 1.27997 w_k1_IBM_s113
                                 + 1.14976 w_k1_INTC_s113
                                 + 1.53268 w_k1_KO_s113 + 1.23082 w_k1_MAT_s113
                                 + 1.23771 w_k1_MSFT_s113
                                 + 1.15273 w_k1_PG_s113 >= 1
 z_k1_s1136_constraint:          var_1 + z_k1_s1136 + 0.896866 w_k1_ADM_s113
                                 + 1.16832 w_k1_AXP_s113
                                 + 1.00294 w_k1_DIS_s113
                                 + 0.961782 w_k1_FDX_s113
                                 + 1.4827 w_k1_IBM_s113
                                 + 1.18803 w_k1_INTC_s113
                                 + 1.17232 w_k1_KO_s113 + 1.51876 w_k1_MAT_s113
                                 + 0.959853 w_k1_MSFT_s113
                                 + 1.02584 w_k1_PG_s113 >= 1
 z_k1_s1137_constraint:          var_1 + z_k1_s1137 + 1.18404 w_k1_ADM_s113
                                 + 1.30727 w_k1_AXP_s113
                                 + 1.25106 w_k1_DIS_s113
                                 + 1.24989 w_k1_FDX_s113
                                 + 0.98692 w_k1_IBM_s113
                                 + 1.40941 w_k1_INTC_s113
                                 + 1.16196 w_k1_KO_s113
                                 + 0.918389 w_k1_MAT_s113
                                 + 1.12302 w_k1_MSFT_s113
                                 + 1.02093 w_k1_PG_s113 >= 1
 z_k1_s1140_constraint:          var_1 + z_k1_s1140 + 0.722056 w_k1_ADM_s114
                                 + 0.682596 w_k1_AXP_s114
                                 + 0.740264 w_k1_DIS_s114
                                 + 0.756227 w_k1_FDX_s114
                                 + 0.740512 w_k1_IBM_s114
                                 + 1.00801 w_k1_INTC_s114
                                 + 0.849855 w_k1_KO_s114
                                 + 0.496895 w_k1_MAT_s114
                                 + 0.854767 w_k1_MSFT_s114
                                 + 0.805727 w_k1_PG_s114 >= 1
 z_k1_s1141_constraint:          var_1 + z_k1_s1141 + 0.620034 w_k1_ADM_s114
                                 + 0.806649 w_k1_AXP_s114
                                 + 0.822462 w_k1_DIS_s114
                                 + 0.698728 w_k1_FDX_s114
                                 + 1.24499 w_k1_IBM_s114
                                 + 1.01199 w_k1_INTC_s114
                                 + 1.19784 w_k1_KO_s114 + 1.00118 w_k1_MAT_s114
                                 + 0.862021 w_k1_MSFT_s114
                                 + 0.870011 w_k1_PG_s114 >= 1
 z_k1_s1142_constraint:          var_1 + z_k1_s1142 + 1.06525 w_k1_ADM_s114
                                 + 1.18185 w_k1_AXP_s114
                                 + 1.39678 w_k1_DIS_s114
                                 + 1.21657 w_k1_FDX_s114
                                 + 1.19606 w_k1_IBM_s114
                                 + 1.23186 w_k1_INTC_s114 + 1.4064 w_k1_KO_s114
                                 + 1.03696 w_k1_MAT_s114
                                 + 1.27295 w_k1_MSFT_s114
                                 + 1.09037 w_k1_PG_s114 >= 1
 z_k1_s1143_constraint:          var_1 + z_k1_s1143 + 0.924813 w_k1_ADM_s114
                                 + 0.994724 w_k1_AXP_s114
                                 + 0.955967 w_k1_DIS_s114
                                 + 0.911163 w_k1_FDX_s114
                                 + 1.15873 w_k1_IBM_s114
                                 + 0.988613 w_k1_INTC_s114
                                 + 1.02592 w_k1_KO_s114 + 1.02794 w_k1_MAT_s114
                                 + 1.07255 w_k1_MSFT_s114
                                 + 0.958377 w_k1_PG_s114 >= 1
 z_k1_s1144_constraint:          var_1 + z_k1_s1144 + 0.831404 w_k1_ADM_s114
                                 + 0.841572 w_k1_AXP_s114
                                 + 0.778965 w_k1_DIS_s114
                                 + 0.938788 w_k1_FDX_s114
                                 + 0.598217 w_k1_IBM_s114
                                 + 0.929382 w_k1_INTC_s114
                                 + 0.735046 w_k1_KO_s114
                                 + 0.584993 w_k1_MAT_s114
                                 + 0.897816 w_k1_MSFT_s114
                                 + 0.853835 w_k1_PG_s114 >= 1
 z_k1_s1145_constraint:          var_1 + z_k1_s1145 + 0.973309 w_k1_ADM_s114
                                 + 1.62293 w_k1_AXP_s114
                                 + 1.75332 w_k1_DIS_s114
                                 + 1.19631 w_k1_FDX_s114 + 1.8341 w_k1_IBM_s114
                                 + 1.14715 w_k1_INTC_s114
                                 + 1.89251 w_k1_KO_s114 + 2.14937 w_k1_MAT_s114
                                 + 1.24728 w_k1_MSFT_s114
                                 + 1.18372 w_k1_PG_s114 >= 1
 z_k1_s1146_constraint:          var_1 + z_k1_s1146 + 0.875568 w_k1_ADM_s114
                                 + 0.936174 w_k1_AXP_s114
                                 + 0.834943 w_k1_DIS_s114
                                 + 0.890225 w_k1_FDX_s114
                                 + 0.868267 w_k1_IBM_s114
                                 + 1.03505 w_k1_INTC_s114
                                 + 0.858521 w_k1_KO_s114
                                 + 0.620166 w_k1_MAT_s114
                                 + 0.912647 w_k1_MSFT_s114
                                 + 0.855581 w_k1_PG_s114 >= 1
 z_k1_s1147_constraint:          var_1 + z_k1_s1147 + 1.07681 w_k1_ADM_s114
                                 + 1.14124 w_k1_AXP_s114
                                 + 1.29849 w_k1_DIS_s114
                                 + 1.15058 w_k1_FDX_s114
                                 + 0.802994 w_k1_IBM_s114
                                 + 1.3049 w_k1_INTC_s114 + 1.08752 w_k1_KO_s114
                                 + 0.619868 w_k1_MAT_s114
                                 + 1.26306 w_k1_MSFT_s114
                                 + 0.984458 w_k1_PG_s114 >= 1
 z_k1_s1150_constraint:          var_1 + z_k1_s1150 + 1.08382 w_k1_ADM_s115
                                 + 1.20666 w_k1_AXP_s115
                                 + 1.13785 w_k1_DIS_s115
                                 + 1.01488 w_k1_FDX_s115
                                 + 1.43704 w_k1_IBM_s115
                                 + 1.06221 w_k1_INTC_s115
                                 + 1.12961 w_k1_KO_s115 + 1.5703 w_k1_MAT_s115
                                 + 1.04097 w_k1_MSFT_s115
                                 + 1.13825 w_k1_PG_s115 >= 1
 z_k1_s1151_constraint:          var_1 + z_k1_s1151 + 1.73921 w_k1_ADM_s115
                                 + 1.73792 w_k1_AXP_s115 + 1.9594 w_k1_DIS_s115
                                 + 1.53897 w_k1_FDX_s115
                                 + 1.44269 w_k1_IBM_s115
                                 + 1.22312 w_k1_INTC_s115
                                 + 1.34848 w_k1_KO_s115 + 1.5052 w_k1_MAT_s115
                                 + 1.29156 w_k1_MSFT_s115
                                 + 1.45592 w_k1_PG_s115 >= 1
 z_k1_s1152_constraint:          var_1 + z_k1_s1152 + 1.63588 w_k1_ADM_s115
                                 + 1.61491 w_k1_AXP_s115
                                 + 1.37595 w_k1_DIS_s115
                                 + 1.46188 w_k1_FDX_s115
                                 + 0.904707 w_k1_IBM_s115
                                 + 1.22744 w_k1_INTC_s115
                                 + 0.991083 w_k1_KO_s115
                                 + 1.12122 w_k1_MAT_s115
                                 + 1.1515 w_k1_MSFT_s115 + 1.14669 w_k1_PG_s115
                                 >= 1
 z_k1_s1153_constraint:          var_1 + z_k1_s1153 + 1.09716 w_k1_ADM_s115
                                 + 1.04291 w_k1_AXP_s115
                                 + 1.18287 w_k1_DIS_s115
                                 + 1.09081 w_k1_FDX_s115
                                 + 0.913835 w_k1_IBM_s115
                                 + 0.91558 w_k1_INTC_s115
                                 + 1.07413 w_k1_KO_s115 + 1.06009 w_k1_MAT_s115
                                 + 0.96384 w_k1_MSFT_s115
                                 + 1.15974 w_k1_PG_s115 >= 1
 z_k1_s1154_constraint:          var_1 + z_k1_s1154 + 1.34378 w_k1_ADM_s115
                                 + 1.2471 w_k1_AXP_s115 + 1.22469 w_k1_DIS_s115
                                 + 1.22672 w_k1_FDX_s115
                                 + 1.33318 w_k1_IBM_s115
                                 + 1.1911 w_k1_INTC_s115 + 1.18499 w_k1_KO_s115
                                 + 1.06988 w_k1_MAT_s115
                                 + 1.17398 w_k1_MSFT_s115
                                 + 1.22501 w_k1_PG_s115 >= 1
 z_k1_s1155_constraint:          var_1 + z_k1_s1155 + 1.08767 w_k1_ADM_s115
                                 + 1.03858 w_k1_AXP_s115
                                 + 0.911628 w_k1_DIS_s115
                                 + 0.913451 w_k1_FDX_s115
                                 + 0.909677 w_k1_IBM_s115
                                 + 0.869472 w_k1_INTC_s115
                                 + 0.850996 w_k1_KO_s115
                                 + 0.866831 w_k1_MAT_s115
                                 + 0.88155 w_k1_MSFT_s115
                                 + 1.01667 w_k1_PG_s115 >= 1
 z_k1_s1156_constraint:          var_1 + z_k1_s1156 + 0.825111 w_k1_ADM_s115
                                 + 0.574053 w_k1_AXP_s115
                                 + 0.646571 w_k1_DIS_s115
                                 + 0.695608 w_k1_FDX_s115
                                 + 0.947374 w_k1_IBM_s115
                                 + 0.877653 w_k1_INTC_s115
                                 + 0.899533 w_k1_KO_s115
                                 + 0.662676 w_k1_MAT_s115
                                 + 0.813652 w_k1_MSFT_s115
                                 + 0.884485 w_k1_PG_s115 >= 1
 z_k1_s1157_constraint:          var_1 + z_k1_s1157 + 1.25297 w_k1_ADM_s115
                                 + 1.14249 w_k1_AXP_s115
                                 + 1.06244 w_k1_DIS_s115 + 1.1714 w_k1_FDX_s115
                                 + 0.770976 w_k1_IBM_s115
                                 + 0.934399 w_k1_INTC_s115
                                 + 0.883451 w_k1_KO_s115
                                 + 0.917548 w_k1_MAT_s115
                                 + 0.901856 w_k1_MSFT_s115
                                 + 1.1077 w_k1_PG_s115 >= 1
 z_k1_s1160_constraint:          var_1 + z_k1_s1160 + 1.523 w_k1_ADM_s116
                                 + 1.68898 w_k1_AXP_s116
                                 + 1.32099 w_k1_DIS_s116
                                 + 1.40736 w_k1_FDX_s116
                                 + 0.815719 w_k1_IBM_s116
                                 + 1.2177 w_k1_INTC_s116
                                 + 0.967833 w_k1_KO_s116
                                 + 1.19309 w_k1_MAT_s116
                                 + 1.18232 w_k1_MSFT_s116
                                 + 1.05206 w_k1_PG_s116 >= 1
 z_k1_s1161_constraint:          var_1 + z_k1_s1161 + 1.48193 w_k1_ADM_s116
                                 + 1.63941 w_k1_AXP_s116
                                 + 1.69318 w_k1_DIS_s116
                                 + 1.42127 w_k1_FDX_s116
                                 + 1.19059 w_k1_IBM_s116
                                 + 1.42887 w_k1_INTC_s116 + 1.3905 w_k1_KO_s116
                                 + 1.31637 w_k1_MAT_s116
                                 + 1.30157 w_k1_MSFT_s116
                                 + 1.34543 w_k1_PG_s116 >= 1
 z_k1_s1162_constraint:          var_1 + z_k1_s1162 + 1.22758 w_k1_ADM_s116
                                 + 1.29776 w_k1_AXP_s116
                                 + 1.18996 w_k1_DIS_s116
                                 + 1.06415 w_k1_FDX_s116
                                 + 1.50957 w_k1_IBM_s116
                                 + 1.12393 w_k1_INTC_s116 + 1.3558 w_k1_KO_s116
                                 + 1.32684 w_k1_MAT_s116
                                 + 0.982421 w_k1_MSFT_s116
                                 + 0.99194 w_k1_PG_s116 >= 1
 z_k1_s1163_constraint:          var_1 + z_k1_s1163 + 0.932794 w_k1_ADM_s116
                                 + 0.769374 w_k1_AXP_s116
                                 + 0.867352 w_k1_DIS_s116
                                 + 0.831087 w_k1_FDX_s116
                                 + 1.00937 w_k1_IBM_s116
                                 + 1.11584 w_k1_INTC_s116
                                 + 0.963154 w_k1_KO_s116
                                 + 0.828478 w_k1_MAT_s116
                                 + 0.902737 w_k1_MSFT_s116
                                 + 0.884425 w_k1_PG_s116 >= 1
 z_k1_s1164_constraint:          var_1 + z_k1_s1164 + 0.999278 w_k1_ADM_s116
                                 + 0.9638 w_k1_AXP_s116 + 1.04588 w_k1_DIS_s116
                                 + 0.997562 w_k1_FDX_s116
                                 + 1.1853 w_k1_IBM_s116
                                 + 1.08134 w_k1_INTC_s116
                                 + 1.13144 w_k1_KO_s116
                                 + 0.738754 w_k1_MAT_s116
                                 + 1.03542 w_k1_MSFT_s116
                                 + 1.05848 w_k1_PG_s116 >= 1
 z_k1_s1165_constraint:          var_1 + z_k1_s1165 + 1.11156 w_k1_ADM_s116
                                 + 0.911634 w_k1_AXP_s116
                                 + 1.1158 w_k1_DIS_s116 + 1.0828 w_k1_FDX_s116
                                 + 0.830019 w_k1_IBM_s116
                                 + 1.28042 w_k1_INTC_s116
                                 + 1.07131 w_k1_KO_s116
                                 + 0.575083 w_k1_MAT_s116
                                 + 1.18606 w_k1_MSFT_s116
                                 + 0.981104 w_k1_PG_s116 >= 1
 z_k1_s1166_constraint:          var_1 + z_k1_s1166 + 0.966647 w_k1_ADM_s116
                                 + 1.45406 w_k1_AXP_s116
                                 + 1.10439 w_k1_DIS_s116
                                 + 1.19303 w_k1_FDX_s116
                                 + 1.15525 w_k1_IBM_s116
                                 + 1.31653 w_k1_INTC_s116
                                 + 0.998322 w_k1_KO_s116
                                 + 1.27319 w_k1_MAT_s116
                                 + 1.0629 w_k1_MSFT_s116
                                 + 0.996363 w_k1_PG_s116 >= 1
 z_k1_s1167_constraint:          var_1 + z_k1_s1167 + 0.748718 w_k1_ADM_s116
                                 + 0.721937 w_k1_AXP_s116
                                 + 0.439916 w_k1_DIS_s116
                                 + 0.695333 w_k1_FDX_s116
                                 + 0.707546 w_k1_IBM_s116
                                 + 0.882564 w_k1_INTC_s116
                                 + 0.677988 w_k1_KO_s116
                                 + 0.502394 w_k1_MAT_s116
                                 + 0.666314 w_k1_MSFT_s116
                                 + 0.672649 w_k1_PG_s116 >= 1
 z_k1_s1170_constraint:          var_1 + z_k1_s1170 + 1.03129 w_k1_ADM_s117
                                 + 1.10191 w_k1_AXP_s117
                                 + 0.971453 w_k1_DIS_s117
                                 + 0.941499 w_k1_FDX_s117
                                 + 1.20797 w_k1_IBM_s117
                                 + 1.12084 w_k1_INTC_s117
                                 + 1.03152 w_k1_KO_s117 + 1.1021 w_k1_MAT_s117
                                 + 0.968308 w_k1_MSFT_s117
                                 + 1.00511 w_k1_PG_s117 >= 1
 z_k1_s1171_constraint:          var_1 + z_k1_s1171 + 0.986894 w_k1_ADM_s117
                                 + 0.70831 w_k1_AXP_s117
                                 + 1.06358 w_k1_DIS_s117
                                 + 0.972963 w_k1_FDX_s117
                                 + 1.01798 w_k1_IBM_s117
                                 + 1.00173 w_k1_INTC_s117
                                 + 1.14109 w_k1_KO_s117
                                 + 0.806427 w_k1_MAT_s117
                                 + 0.97265 w_k1_MSFT_s117
                                 + 0.965125 w_k1_PG_s117 >= 1
 z_k1_s1172_constraint:          var_1 + z_k1_s1172 + 0.957539 w_k1_ADM_s117
                                 + 0.947656 w_k1_AXP_s117
                                 + 0.79432 w_k1_DIS_s117
                                 + 0.910989 w_k1_FDX_s117
                                 + 0.820305 w_k1_IBM_s117
                                 + 0.83961 w_k1_INTC_s117
                                 + 0.861155 w_k1_KO_s117
                                 + 0.798845 w_k1_MAT_s117
                                 + 0.851734 w_k1_MSFT_s117
                                 + 0.861376 w_k1_PG_s117 >= 1
 z_k1_s1173_constraint:          var_1 + z_k1_s1173 + 1.20505 w_k1_ADM_s117
                                 + 1.22374 w_k1_AXP_s117
                                 + 1.23538 w_k1_DIS_s117
                                 + 1.29768 w_k1_FDX_s117
                                 + 1.08386 w_k1_IBM_s117
                                 + 1.31699 w_k1_INTC_s117
                                 + 1.34056 w_k1_KO_s117 + 0.90106 w_k1_MAT_s117
                                 + 1.19403 w_k1_MSFT_s117
                                 + 1.21853 w_k1_PG_s117 >= 1
 z_k1_s1174_constraint:          var_1 + z_k1_s1174 + 0.906853 w_k1_ADM_s117
                                 + 1.21583 w_k1_AXP_s117
                                 + 0.883074 w_k1_DIS_s117
                                 + 0.966781 w_k1_FDX_s117
                                 + 0.995381 w_k1_IBM_s117
                                 + 1.07268 w_k1_INTC_s117
                                 + 1.00085 w_k1_KO_s117
                                 + 0.884284 w_k1_MAT_s117
                                 + 0.947749 w_k1_MSFT_s117
                                 + 1.08671 w_k1_PG_s117 >= 1
 z_k1_s1175_constraint:          var_1 + z_k1_s1175 + 0.847304 w_k1_ADM_s117
                                 + 1.09779 w_k1_AXP_s117
                                 + 1.09991 w_k1_DIS_s117
                                 + 0.944989 w_k1_FDX_s117
                                 + 1.45759 w_k1_IBM_s117
                                 + 0.922317 w_k1_INTC_s117
                                 + 1.1653 w_k1_KO_s117 + 1.15968 w_k1_MAT_s117
                                 + 0.861345 w_k1_MSFT_s117
                                 + 1.03306 w_k1_PG_s117 >= 1
 z_k1_s1176_constraint:          var_1 + z_k1_s1176 + 1.25893 w_k1_ADM_s117
                                 + 1.14167 w_k1_AXP_s117
                                 + 0.969192 w_k1_DIS_s117
                                 + 1.10453 w_k1_FDX_s117
                                 + 0.663204 w_k1_IBM_s117
                                 + 1.01784 w_k1_INTC_s117
                                 + 0.882892 w_k1_KO_s117
                                 + 0.775984 w_k1_MAT_s117
                                 + 1.01106 w_k1_MSFT_s117
                                 + 0.90679 w_k1_PG_s117 >= 1
 z_k1_s1177_constraint:          var_1 + z_k1_s1177 + 0.940014 w_k1_ADM_s117
                                 + 0.7935 w_k1_AXP_s117
                                 + 0.916823 w_k1_DIS_s117
                                 + 1.01566 w_k1_FDX_s117
                                 + 0.930342 w_k1_IBM_s117
                                 + 0.94423 w_k1_INTC_s117 + 0.9019 w_k1_KO_s117
                                 + 0.611423 w_k1_MAT_s117
                                 + 0.988074 w_k1_MSFT_s117
                                 + 0.956489 w_k1_PG_s117 >= 1
 k1_return_constraint:           0.0664900245880653 w_k1_ADM_s000
                                 + 0.0276708389858245 w_k1_ADM_s001
                                 + 0.0330632498169778 w_k1_ADM_s002
                                 + 0.0304167002384332 w_k1_ADM_s003
                                 + 0.0277007871158213 w_k1_ADM_s004
                                 + 0.0182497735816145 w_k1_ADM_s005
                                 + 0.0372244027303365 w_k1_ADM_s006
                                 + 0.0745669982556326 w_k1_ADM_s007
                                 + 0.0171421485545563 w_k1_ADM_s010
                                 + 0.0320215830465691 w_k1_ADM_s011
                                 + 0.0390187778021715 w_k1_ADM_s012
                                 + 0.0233889383683612 w_k1_ADM_s013
                                 + 0.0434621629546755 w_k1_ADM_s014
                                 + 0.0450574424685744 w_k1_ADM_s015
                                 + 0.0157072051678738 w_k1_ADM_s016
                                 + 0.015425088124213 w_k1_ADM_s017
                                 + 0.0572667130608939 w_k1_ADM_s100
                                 + 0.021495095271719 w_k1_ADM_s101
                                 + 0.0132823304007355 w_k1_ADM_s102
                                 + 0.0537005745489602 w_k1_ADM_s103
                                 + 0.0261226325719687 w_k1_ADM_s104
                                 + 0.0120968082217851 w_k1_ADM_s105
                                 + 0.0317640242711852 w_k1_ADM_s106
                                 + 0.0183542907292037 w_k1_ADM_s107
                                 + 0.0321368309257009 w_k1_ADM_s110
                                 + 0.02547793170335 w_k1_ADM_s111
                                 + 0.0149325584320987 w_k1_ADM_s112
                                 + 0.0354067838363165 w_k1_ADM_s113
                                 + 0.0479100296872078 w_k1_ADM_s114
                                 + 0.0178364988668212 w_k1_ADM_s115
                                 + 0.0237513308975676 w_k1_ADM_s116
                                 + 0.0143933968536633 w_k1_ADM_s117
                                 + 0.0695709182864949 w_k1_AXP_s000
                                 + 0.0251771529569461 w_k1_AXP_s001
                                 + 0.0332586941219916 w_k1_AXP_s002
                                 + 0.0308666148593702 w_k1_AXP_s003
                                 + 0.0281416519233631 w_k1_AXP_s004
                                 + 0.0216369526098589 w_k1_AXP_s005
                                 + 0.0353579349409623 w_k1_AXP_s006
                                 + 0.0806978414517214 w_k1_AXP_s007
                                 + 0.0180993101028615 w_k1_AXP_s010
                                 + 0.0299891673847253 w_k1_AXP_s011
                                 + 0.0415984304416336 w_k1_AXP_s012
                                 + 0.0197485903557883 w_k1_AXP_s013
                                 + 0.0436047950645331 w_k1_AXP_s014
                                 + 0.0449002282352417 w_k1_AXP_s015
                                 + 0.0149979931405356 w_k1_AXP_s016
                                 + 0.0184614214166536 w_k1_AXP_s017
                                 + 0.0556259253150838 w_k1_AXP_s100
                                 + 0.0228938077742485 w_k1_AXP_s101
                                 + 0.016799306161825 w_k1_AXP_s102
                                 + 0.0533185318915886 w_k1_AXP_s103
                                 + 0.0277945145353529 w_k1_AXP_s104
                                 + 0.0132021763816196 w_k1_AXP_s105
                                 + 0.0298541571999463 w_k1_AXP_s106
                                 + 0.018901661966901 w_k1_AXP_s107
                                 + 0.0326216310454699 w_k1_AXP_s110
                                 + 0.0320197961098008 w_k1_AXP_s111
                                 + 0.0177228158601578 w_k1_AXP_s112
                                 + 0.0414095637203886 w_k1_AXP_s113
                                 + 0.0540237843702706 w_k1_AXP_s114
                                 + 0.0167119296110081 w_k1_AXP_s115
                                 + 0.024166964045369 w_k1_AXP_s116
                                 + 0.0150442755049823 w_k1_AXP_s117
                                 + 0.0653361642505696 w_k1_DIS_s000
                                 + 0.0255624704365868 w_k1_DIS_s001
                                 + 0.0362788936206124 w_k1_DIS_s002
                                 + 0.0310663842953706 w_k1_DIS_s003
                                 + 0.0293249136950562 w_k1_DIS_s004
                                 + 0.0206682228759931 w_k1_DIS_s005
                                 + 0.0346472110139904 w_k1_DIS_s006
                                 + 0.0843565072897636 w_k1_DIS_s007
                                 + 0.0199048046981642 w_k1_DIS_s010
                                 + 0.0295600651309006 w_k1_DIS_s011
                                 + 0.0414929573343612 w_k1_DIS_s012
                                 + 0.0236311780848972 w_k1_DIS_s013
                                 + 0.0414313798971415 w_k1_DIS_s014
                                 + 0.0432911863482346 w_k1_DIS_s015
                                 + 0.0138364057606184 w_k1_DIS_s016
                                 + 0.0190772209205339 w_k1_DIS_s017
                                 + 0.0584708911624747 w_k1_DIS_s100
                                 + 0.0228329446296791 w_k1_DIS_s101
                                 + 0.0168859151434742 w_k1_DIS_s102
                                 + 0.0543855865377233 w_k1_DIS_s103
                                 + 0.029384136683452 w_k1_DIS_s104
                                 + 0.0146939060513921 w_k1_DIS_s105
                                 + 0.0303202633165459 w_k1_DIS_s106
                                 + 0.0203279931637902 w_k1_DIS_s107
                                 + 0.0331358795063522 w_k1_DIS_s110
                                 + 0.0323777469536216 w_k1_DIS_s111
                                 + 0.0173715626379394 w_k1_DIS_s112
                                 + 0.0401532149796225 w_k1_DIS_s113
                                 + 0.0573023563875898 w_k1_DIS_s114
                                 + 0.0163063856065751 w_k1_DIS_s115
                                 + 0.0230080844250684 w_k1_DIS_s116
                                 + 0.0139751438628311 w_k1_DIS_s117
                                 + 0.0672209543038769 w_k1_FDX_s000
                                 + 0.0259532408979061 w_k1_FDX_s001
                                 + 0.0358168573481012 w_k1_FDX_s002
                                 + 0.0306751181212772 w_k1_FDX_s003
                                 + 0.0290861619080286 w_k1_FDX_s004
                                 + 0.0226580463565818 w_k1_FDX_s005
                                 + 0.036961516435573 w_k1_FDX_s006
                                 + 0.0808929872998174 w_k1_FDX_s007
                                 + 0.0182078821429844 w_k1_FDX_s010
                                 + 0.0295812483950703 w_k1_FDX_s011
                                 + 0.0423254611886485 w_k1_FDX_s012
                                 + 0.0208018609689553 w_k1_FDX_s013
                                 + 0.0444446481235136 w_k1_FDX_s014
                                 + 0.0422039939130243 w_k1_FDX_s015
                                 + 0.0140372303641277 w_k1_FDX_s016
                                 + 0.0177171697081612 w_k1_FDX_s017
                                 + 0.0564444646109459 w_k1_FDX_s100
                                 + 0.0235368797413079 w_k1_FDX_s101
                                 + 0.0152378383505991 w_k1_FDX_s102
                                 + 0.0560239178471661 w_k1_FDX_s103
                                 + 0.025893448797127 w_k1_FDX_s104
                                 + 0.0130459077720648 w_k1_FDX_s105
                                 + 0.0307552049419522 w_k1_FDX_s106
                                 + 0.0200427966132873 w_k1_FDX_s107
                                 + 0.0306590215736822 w_k1_FDX_s110
                                 + 0.031911796918691 w_k1_FDX_s111
                                 + 0.0155561463448625 w_k1_FDX_s112
                                 + 0.0393625672089394 w_k1_FDX_s113
                                 + 0.0518630546105281 w_k1_FDX_s114
                                 + 0.015914840768356 w_k1_FDX_s115
                                 + 0.0228364177382868 w_k1_FDX_s116
                                 + 0.0142638377050589 w_k1_FDX_s117
                                 + 0.0672873307122794 w_k1_IBM_s000
                                 + 0.0261307703736928 w_k1_IBM_s001
                                 + 0.0351589591795711 w_k1_IBM_s002
                                 + 0.0298046902922562 w_k1_IBM_s003
                                 + 0.0300256560319939 w_k1_IBM_s004
                                 + 0.0218618762345021 w_k1_IBM_s005
                                 + 0.0387275273982547 w_k1_IBM_s006
                                 + 0.0846695022149025 w_k1_IBM_s007
                                 + 0.0189942885295887 w_k1_IBM_s010
                                 + 0.0301774352351193 w_k1_IBM_s011
                                 + 0.0423463881268858 w_k1_IBM_s012
                                 + 0.0217521780859982 w_k1_IBM_s013
                                 + 0.0428520425537456 w_k1_IBM_s014
                                 + 0.040586352333894 w_k1_IBM_s015
                                 + 0.0133488651655735 w_k1_IBM_s016
                                 + 0.0182977932087553 w_k1_IBM_s017
                                 + 0.0553305626289309 w_k1_IBM_s100
                                 + 0.0233652371196179 w_k1_IBM_s101
                                 + 0.0154402554943424 w_k1_IBM_s102
                                 + 0.0543966477698018 w_k1_IBM_s103
                                 + 0.0264664167630937 w_k1_IBM_s104
                                 + 0.0135939128864037 w_k1_IBM_s105
                                 + 0.031084426649395 w_k1_IBM_s106
                                 + 0.0207131018720326 w_k1_IBM_s107
                                 + 0.0310581811345705 w_k1_IBM_s110
                                 + 0.0309590372973104 w_k1_IBM_s111
                                 + 0.0157042930861549 w_k1_IBM_s112
                                 + 0.0414553949416231 w_k1_IBM_s113
                                 + 0.0559558512768235 w_k1_IBM_s114
                                 + 0.0159022029795797 w_k1_IBM_s115
                                 + 0.0229834044301531 w_k1_IBM_s116
                                 + 0.0151639145440353 w_k1_IBM_s117
                                 + 0.0759874165968399 w_k1_INTC_s000
                                 + 0.0250387680471659 w_k1_INTC_s001
                                 + 0.0363665792960864 w_k1_INTC_s002
                                 + 0.0350782360608904 w_k1_INTC_s003
                                 + 0.0348478062542121 w_k1_INTC_s004
                                 + 0.0241899465306742 w_k1_INTC_s005
                                 + 0.0360623110657986 w_k1_INTC_s006
                                 + 0.0834502730573315 w_k1_INTC_s007
                                 + 0.0175802446109031 w_k1_INTC_s010
                                 + 0.0281472086912501 w_k1_INTC_s011
                                 + 0.0362804738281369 w_k1_INTC_s012
                                 + 0.0202466360240894 w_k1_INTC_s013
                                 + 0.0419057476268345 w_k1_INTC_s014
                                 + 0.0499061713756337 w_k1_INTC_s015
                                 + 0.0144705574220594 w_k1_INTC_s016
                                 + 0.0178327740643231 w_k1_INTC_s017
                                 + 0.0540225252398216 w_k1_INTC_s100
                                 + 0.0246366837221026 w_k1_INTC_s101
                                 + 0.0156387999759809 w_k1_INTC_s102
                                 + 0.0562066115638356 w_k1_INTC_s103
                                 + 0.0250289571269468 w_k1_INTC_s104
                                 + 0.0136182293108512 w_k1_INTC_s105
                                 + 0.0324452806080336 w_k1_INTC_s106
                                 + 0.0189645275113309 w_k1_INTC_s107
                                 + 0.0312473960084808 w_k1_INTC_s110
                                 + 0.0286559996529466 w_k1_INTC_s111
                                 + 0.0177678251708672 w_k1_INTC_s112
                                 + 0.0431523832691395 w_k1_INTC_s113
                                 + 0.0583432814697537 w_k1_INTC_s114
                                 + 0.0150660481288213 w_k1_INTC_s115
                                 + 0.0258816490412908 w_k1_INTC_s116
                                 + 0.0147634690486043 w_k1_INTC_s117
                                 + 0.0655661753303764 w_k1_KO_s000
                                 + 0.0265701575882504 w_k1_KO_s001
                                 + 0.033552753882693 w_k1_KO_s002
                                 + 0.0312980678042107 w_k1_KO_s003
                                 + 0.0261440636755246 w_k1_KO_s004
                                 + 0.0205858036948319 w_k1_KO_s005
                                 + 0.0361246534599204 w_k1_KO_s006
                                 + 0.0831403972147008 w_k1_KO_s007
                                 + 0.0186796387349904 w_k1_KO_s010
                                 + 0.0311123900150652 w_k1_KO_s011
                                 + 0.0417401307975447 w_k1_KO_s012
                                 + 0.0215384962997177 w_k1_KO_s013
                                 + 0.0435851247525601 w_k1_KO_s014
                                 + 0.0456959635888239 w_k1_KO_s015
                                 + 0.0140718094863946 w_k1_KO_s016
                                 + 0.016336683085836 w_k1_KO_s017
                                 + 0.0561211870797842 w_k1_KO_s100
                                 + 0.0239237802657874 w_k1_KO_s101
                                 + 0.0152580576817808 w_k1_KO_s102
                                 + 0.0542313225414983 w_k1_KO_s103
                                 + 0.0289604223298636 w_k1_KO_s104
                                 + 0.0143095422186251 w_k1_KO_s105
                                 + 0.0304716349734186 w_k1_KO_s106
                                 + 0.0209005104088558 w_k1_KO_s107
                                 + 0.0289910329094119 w_k1_KO_s110
                                 + 0.0319023540432401 w_k1_KO_s111
                                 + 0.0177402597233022 w_k1_KO_s112
                                 + 0.0443302285759258 w_k1_KO_s113
                                 + 0.060122374611159 w_k1_KO_s114
                                 + 0.015238578821699 w_k1_KO_s115
                                 + 0.0232210333529805 w_k1_KO_s116
                                 + 0.0148710857211989 w_k1_KO_s117
                                 + 0.0576256078763792 w_k1_MAT_s000
                                 + 0.0266445113900605 w_k1_MAT_s001
                                 + 0.0290649853228403 w_k1_MAT_s002
                                 + 0.0257123188444298 w_k1_MAT_s003
                                 + 0.0218633870729943 w_k1_MAT_s004
                                 + 0.021491266427452 w_k1_MAT_s005
                                 + 0.0326086678681516 w_k1_MAT_s006
                                 + 0.074134980940347 w_k1_MAT_s007
                                 + 0.0229827585205233 w_k1_MAT_s010
                                 + 0.0297941831808343 w_k1_MAT_s011
                                 + 0.0417708305670888 w_k1_MAT_s012
                                 + 0.0235593154153695 w_k1_MAT_s013
                                 + 0.0416762747886387 w_k1_MAT_s014
                                 + 0.0417497369014921 w_k1_MAT_s015
                                 + 0.012442913495218 w_k1_MAT_s016
                                 + 0.0172564313055004 w_k1_MAT_s017
                                 + 0.0656129405324473 w_k1_MAT_s100
                                 + 0.0203086636257801 w_k1_MAT_s101
                                 + 0.0154532125715539 w_k1_MAT_s102
                                 + 0.0534531454376213 w_k1_MAT_s103
                                 + 0.0239219116117788 w_k1_MAT_s104
                                 + 0.0159230594914114 w_k1_MAT_s105
                                 + 0.0329886611630927 w_k1_MAT_s106
                                 + 0.0186877440262093 w_k1_MAT_s107
                                 + 0.0348155119037611 w_k1_MAT_s110
                                 + 0.0288109729991972 w_k1_MAT_s111
                                 + 0.0200615088335754 w_k1_MAT_s112
                                 + 0.040445657770295 w_k1_MAT_s113
                                 + 0.0485238727410137 w_k1_MAT_s114
                                 + 0.01548344742811 w_k1_MAT_s115
                                 + 0.0204386296859466 w_k1_MAT_s116
                                 + 0.0133208447881498 w_k1_MAT_s117
                                 + 0.0703087092946166 w_k1_MSFT_s000
                                 + 0.0270486087501504 w_k1_MSFT_s001
                                 + 0.0345340393412972 w_k1_MSFT_s002
                                 + 0.0320331344576238 w_k1_MSFT_s003
                                 + 0.0285314398848063 w_k1_MSFT_s004
                                 + 0.023310882532723 w_k1_MSFT_s005
                                 + 0.0366579173003202 w_k1_MSFT_s006
                                 + 0.0776620829558266 w_k1_MSFT_s007
                                 + 0.0165346151640286 w_k1_MSFT_s010
                                 + 0.0306137352197236 w_k1_MSFT_s011
                                 + 0.040411955449028 w_k1_MSFT_s012
                                 + 0.022593823431273 w_k1_MSFT_s013
                                 + 0.0416104801710444 w_k1_MSFT_s014
                                 + 0.0422358452307671 w_k1_MSFT_s015
                                 + 0.0128079228438735 w_k1_MSFT_s016
                                 + 0.019901920914871 w_k1_MSFT_s017
                                 + 0.0570744020538601 w_k1_MSFT_s100
                                 + 0.0247930446995314 w_k1_MSFT_s101
                                 + 0.0150333692311535 w_k1_MSFT_s102
                                 + 0.0548428790145067 w_k1_MSFT_s103
                                 + 0.0277398264573759 w_k1_MSFT_s104
                                 + 0.0127689425915233 w_k1_MSFT_s105
                                 + 0.0329721328405284 w_k1_MSFT_s106
                                 + 0.0203119294872821 w_k1_MSFT_s107
                                 + 0.0324347287279021 w_k1_MSFT_s110
                                 + 0.0284448356517236 w_k1_MSFT_s111
                                 + 0.0166787305582785 w_k1_MSFT_s112
                                 + 0.0397912824494294 w_k1_MSFT_s113
                                 + 0.0567491953815611 w_k1_MSFT_s114
                                 + 0.0149239547289613 w_k1_MSFT_s115
                                 + 0.0223769743849183 w_k1_MSFT_s116
                                 + 0.0137371122379265 w_k1_MSFT_s117
                                 + 0.0657466062366378 w_k1_PG_s000
                                 + 0.0265533073154125 w_k1_PG_s001
                                 + 0.0344814080007879 w_k1_PG_s002
                                 + 0.0288042451620946 w_k1_PG_s003
                                 + 0.0295430613716798 w_k1_PG_s004
                                 + 0.0236760400542883 w_k1_PG_s005
                                 + 0.0400879508281184 w_k1_PG_s006
                                 + 0.0857354849281934 w_k1_PG_s007
                                 + 0.0181115225202722 w_k1_PG_s010
                                 + 0.0320471121462067 w_k1_PG_s011
                                 + 0.0457970160279213 w_k1_PG_s012
                                 + 0.0229921314004439 w_k1_PG_s013
                                 + 0.0434962518564744 w_k1_PG_s014
                                 + 0.0419296068065114 w_k1_PG_s015
                                 + 0.0143874519527915 w_k1_PG_s016
                                 + 0.0168659356410752 w_k1_PG_s017
                                 + 0.0562786721242554 w_k1_PG_s100
                                 + 0.0220514085010515 w_k1_PG_s101
                                 + 0.0148815404402863 w_k1_PG_s102
                                 + 0.0504332535697826 w_k1_PG_s103
                                 + 0.025995562596097 w_k1_PG_s104
                                 + 0.0148587163263786 w_k1_PG_s105
                                 + 0.0305966026193044 w_k1_PG_s106
                                 + 0.0203904625174387 w_k1_PG_s107
                                 + 0.029836845720176 w_k1_PG_s110
                                 + 0.0309545608134959 w_k1_PG_s111
                                 + 0.0153495488050557 w_k1_PG_s112
                                 + 0.0376077173885352 w_k1_PG_s113
                                 + 0.0508146339032127 w_k1_PG_s114
                                 + 0.0167390927794539 w_k1_PG_s115
                                 + 0.0213888331852561 w_k1_PG_s116
                                 + 0.0143393494244259 w_k1_PG_s117 >= 1.18
Bounds
 0 <= w_0_ADM <= 1
 0 <= w_0_AXP <= 1
 0 <= w_0_DIS <= 1
 0 <= w_0_FDX <= 1
 0 <= w_0_IBM <= 1
 0 <= w_0_INTC <= 1
 0 <= w_0_KO <= 1
 0 <= w_0_MAT <= 1
 0 <= w_0_MSFT <= 1
 0 <= w_0_PG <= 1
 0 <= w_k1_ADM_s0 <= 1
 0 <= w_k1_ADM_s1 <= 1
 0 <= w_k1_ADM_s00 <= 1
 0 <= w_k1_ADM_s01 <= 1
 0 <= w_k1_ADM_s10 <= 1
 0 <= w_k1_ADM_s11 <= 1
 0 <= w_k1_ADM_s000 <= 1
 0 <= w_k1_ADM_s001 <= 1
 0 <= w_k1_ADM_s002 <= 1
 0 <= w_k1_ADM_s003 <= 1
 0 <= w_k1_ADM_s004 <= 1
 0 <= w_k1_ADM_s005 <= 1
 0 <= w_k1_ADM_s006 <= 1
 0 <= w_k1_ADM_s007 <= 1
 0 <= w_k1_ADM_s010 <= 1
 0 <= w_k1_ADM_s011 <= 1
 0 <= w_k1_ADM_s012 <= 1
 0 <= w_k1_ADM_s013 <= 1
 0 <= w_k1_ADM_s014 <= 1
 0 <= w_k1_ADM_s015 <= 1
 0 <= w_k1_ADM_s016 <= 1
 0 <= w_k1_ADM_s017 <= 1
 0 <= w_k1_ADM_s100 <= 1
 0 <= w_k1_ADM_s101 <= 1
 0 <= w_k1_ADM_s102 <= 1
 0 <= w_k1_ADM_s103 <= 1
 0 <= w_k1_ADM_s104 <= 1
 0 <= w_k1_ADM_s105 <= 1
 0 <= w_k1_ADM_s106 <= 1
 0 <= w_k1_ADM_s107 <= 1
 0 <= w_k1_ADM_s110 <= 1
 0 <= w_k1_ADM_s111 <= 1
 0 <= w_k1_ADM_s112 <= 1
 0 <= w_k1_ADM_s113 <= 1
 0 <= w_k1_ADM_s114 <= 1
 0 <= w_k1_ADM_s115 <= 1
 0 <= w_k1_ADM_s116 <= 1
 0 <= w_k1_ADM_s117 <= 1
 0 <= w_k1_AXP_s0 <= 1
 0 <= w_k1_AXP_s1 <= 1
 0 <= w_k1_AXP_s00 <= 1
 0 <= w_k1_AXP_s01 <= 1
 0 <= w_k1_AXP_s10 <= 1
 0 <= w_k1_AXP_s11 <= 1
 0 <= w_k1_AXP_s000 <= 1
 0 <= w_k1_AXP_s001 <= 1
 0 <= w_k1_AXP_s002 <= 1
 0 <= w_k1_AXP_s003 <= 1
 0 <= w_k1_AXP_s004 <= 1
 0 <= w_k1_AXP_s005 <= 1
 0 <= w_k1_AXP_s006 <= 1
 0 <= w_k1_AXP_s007 <= 1
 0 <= w_k1_AXP_s010 <= 1
 0 <= w_k1_AXP_s011 <= 1
 0 <= w_k1_AXP_s012 <= 1
 0 <= w_k1_AXP_s013 <= 1
 0 <= w_k1_AXP_s014 <= 1
 0 <= w_k1_AXP_s015 <= 1
 0 <= w_k1_AXP_s016 <= 1
 0 <= w_k1_AXP_s017 <= 1
 0 <= w_k1_AXP_s100 <= 1
 0 <= w_k1_AXP_s101 <= 1
 0 <= w_k1_AXP_s102 <= 1
 0 <= w_k1_AXP_s103 <= 1
 0 <= w_k1_AXP_s104 <= 1
 0 <= w_k1_AXP_s105 <= 1
 0 <= w_k1_AXP_s106 <= 1
 0 <= w_k1_AXP_s107 <= 1
 0 <= w_k1_AXP_s110 <= 1
 0 <= w_k1_AXP_s111 <= 1
 0 <= w_k1_AXP_s112 <= 1
 0 <= w_k1_AXP_s113 <= 1
 0 <= w_k1_AXP_s114 <= 1
 0 <= w_k1_AXP_s115 <= 1
 0 <= w_k1_AXP_s116 <= 1
 0 <= w_k1_AXP_s117 <= 1
 0 <= w_k1_DIS_s0 <= 1
 0 <= w_k1_DIS_s1 <= 1
 0 <= w_k1_DIS_s00 <= 1
 0 <= w_k1_DIS_s01 <= 1
 0 <= w_k1_DIS_s10 <= 1
 0 <= w_k1_DIS_s11 <= 1
 0 <= w_k1_DIS_s000 <= 1
 0 <= w_k1_DIS_s001 <= 1
 0 <= w_k1_DIS_s002 <= 1
 0 <= w_k1_DIS_s003 <= 1
 0 <= w_k1_DIS_s004 <= 1
 0 <= w_k1_DIS_s005 <= 1
 0 <= w_k1_DIS_s006 <= 1
 0 <= w_k1_DIS_s007 <= 1
 0 <= w_k1_DIS_s010 <= 1
 0 <= w_k1_DIS_s011 <= 1
 0 <= w_k1_DIS_s012 <= 1
 0 <= w_k1_DIS_s013 <= 1
 0 <= w_k1_DIS_s014 <= 1
 0 <= w_k1_DIS_s015 <= 1
 0 <= w_k1_DIS_s016 <= 1
 0 <= w_k1_DIS_s017 <= 1
 0 <= w_k1_DIS_s100 <= 1
 0 <= w_k1_DIS_s101 <= 1
 0 <= w_k1_DIS_s102 <= 1
 0 <= w_k1_DIS_s103 <= 1
 0 <= w_k1_DIS_s104 <= 1
 0 <= w_k1_DIS_s105 <= 1
 0 <= w_k1_DIS_s106 <= 1
 0 <= w_k1_DIS_s107 <= 1
 0 <= w_k1_DIS_s110 <= 1
 0 <= w_k1_DIS_s111 <= 1
 0 <= w_k1_DIS_s112 <= 1
 0 <= w_k1_DIS_s113 <= 1
 0 <= w_k1_DIS_s114 <= 1
 0 <= w_k1_DIS_s115 <= 1
 0 <= w_k1_DIS_s116 <= 1
 0 <= w_k1_DIS_s117 <= 1
 0 <= w_k1_FDX_s0 <= 1
 0 <= w_k1_FDX_s1 <= 1
 0 <= w_k1_FDX_s00 <= 1
 0 <= w_k1_FDX_s01 <= 1
 0 <= w_k1_FDX_s10 <= 1
 0 <= w_k1_FDX_s11 <= 1
 0 <= w_k1_FDX_s000 <= 1
 0 <= w_k1_FDX_s001 <= 1
 0 <= w_k1_FDX_s002 <= 1
 0 <= w_k1_FDX_s003 <= 1
 0 <= w_k1_FDX_s004 <= 1
 0 <= w_k1_FDX_s005 <= 1
 0 <= w_k1_FDX_s006 <= 1
 0 <= w_k1_FDX_s007 <= 1
 0 <= w_k1_FDX_s010 <= 1
 0 <= w_k1_FDX_s011 <= 1
 0 <= w_k1_FDX_s012 <= 1
 0 <= w_k1_FDX_s013 <= 1
 0 <= w_k1_FDX_s014 <= 1
 0 <= w_k1_FDX_s015 <= 1
 0 <= w_k1_FDX_s016 <= 1
 0 <= w_k1_FDX_s017 <= 1
 0 <= w_k1_FDX_s100 <= 1
 0 <= w_k1_FDX_s101 <= 1
 0 <= w_k1_FDX_s102 <= 1
 0 <= w_k1_FDX_s103 <= 1
 0 <= w_k1_FDX_s104 <= 1
 0 <= w_k1_FDX_s105 <= 1
 0 <= w_k1_FDX_s106 <= 1
 0 <= w_k1_FDX_s107 <= 1
 0 <= w_k1_FDX_s110 <= 1
 0 <= w_k1_FDX_s111 <= 1
 0 <= w_k1_FDX_s112 <= 1
 0 <= w_k1_FDX_s113 <= 1
 0 <= w_k1_FDX_s114 <= 1
 0 <= w_k1_FDX_s115 <= 1
 0 <= w_k1_FDX_s116 <= 1
 0 <= w_k1_FDX_s117 <= 1
 0 <= w_k1_IBM_s0 <= 1
 0 <= w_k1_IBM_s1 <= 1
 0 <= w_k1_IBM_s00 <= 1
 0 <= w_k1_IBM_s01 <= 1
 0 <= w_k1_IBM_s10 <= 1
 0 <= w_k1_IBM_s11 <= 1
 0 <= w_k1_IBM_s000 <= 1
 0 <= w_k1_IBM_s001 <= 1
 0 <= w_k1_IBM_s002 <= 1
 0 <= w_k1_IBM_s003 <= 1
 0 <= w_k1_IBM_s004 <= 1
 0 <= w_k1_IBM_s005 <= 1
 0 <= w_k1_IBM_s006 <= 1
 0 <= w_k1_IBM_s007 <= 1
 0 <= w_k1_IBM_s010 <= 1
 0 <= w_k1_IBM_s011 <= 1
 0 <= w_k1_IBM_s012 <= 1
 0 <= w_k1_IBM_s013 <= 1
 0 <= w_k1_IBM_s014 <= 1
 0 <= w_k1_IBM_s015 <= 1
 0 <= w_k1_IBM_s016 <= 1
 0 <= w_k1_IBM_s017 <= 1
 0 <= w_k1_IBM_s100 <= 1
 0 <= w_k1_IBM_s101 <= 1
 0 <= w_k1_IBM_s102 <= 1
 0 <= w_k1_IBM_s103 <= 1
 0 <= w_k1_IBM_s104 <= 1
 0 <= w_k1_IBM_s105 <= 1
 0 <= w_k1_IBM_s106 <= 1
 0 <= w_k1_IBM_s107 <= 1
 0 <= w_k1_IBM_s110 <= 1
 0 <= w_k1_IBM_s111 <= 1
 0 <= w_k1_IBM_s112 <= 1
 0 <= w_k1_IBM_s113 <= 1
 0 <= w_k1_IBM_s114 <= 1
 0 <= w_k1_IBM_s115 <= 1
 0 <= w_k1_IBM_s116 <= 1
 0 <= w_k1_IBM_s117 <= 1
 0 <= w_k1_INTC_s0 <= 1
 0 <= w_k1_INTC_s1 <= 1
 0 <= w_k1_INTC_s00 <= 1
 0 <= w_k1_INTC_s01 <= 1
 0 <= w_k1_INTC_s10 <= 1
 0 <= w_k1_INTC_s11 <= 1
 0 <= w_k1_INTC_s000 <= 1
 0 <= w_k1_INTC_s001 <= 1
 0 <= w_k1_INTC_s002 <= 1
 0 <= w_k1_INTC_s003 <= 1
 0 <= w_k1_INTC_s004 <= 1
 0 <= w_k1_INTC_s005 <= 1
 0 <= w_k1_INTC_s006 <= 1
 0 <= w_k1_INTC_s007 <= 1
 0 <= w_k1_INTC_s010 <= 1
 0 <= w_k1_INTC_s011 <= 1
 0 <= w_k1_INTC_s012 <= 1
 0 <= w_k1_INTC_s013 <= 1
 0 <= w_k1_INTC_s014 <= 1
 0 <= w_k1_INTC_s015 <= 1
 0 <= w_k1_INTC_s016 <= 1
 0 <= w_k1_INTC_s017 <= 1
 0 <= w_k1_INTC_s100 <= 1
 0 <= w_k1_INTC_s101 <= 1
 0 <= w_k1_INTC_s102 <= 1
 0 <= w_k1_INTC_s103 <= 1
 0 <= w_k1_INTC_s104 <= 1
 0 <= w_k1_INTC_s105 <= 1
 0 <= w_k1_INTC_s106 <= 1
 0 <= w_k1_INTC_s107 <= 1
 0 <= w_k1_INTC_s110 <= 1
 0 <= w_k1_INTC_s111 <= 1
 0 <= w_k1_INTC_s112 <= 1
 0 <= w_k1_INTC_s113 <= 1
 0 <= w_k1_INTC_s114 <= 1
 0 <= w_k1_INTC_s115 <= 1
 0 <= w_k1_INTC_s116 <= 1
 0 <= w_k1_INTC_s117 <= 1
 0 <= w_k1_KO_s0 <= 1
 0 <= w_k1_KO_s1 <= 1
 0 <= w_k1_KO_s00 <= 1
 0 <= w_k1_KO_s01 <= 1
 0 <= w_k1_KO_s10 <= 1
 0 <= w_k1_KO_s11 <= 1
 0 <= w_k1_KO_s000 <= 1
 0 <= w_k1_KO_s001 <= 1
 0 <= w_k1_KO_s002 <= 1
 0 <= w_k1_KO_s003 <= 1
 0 <= w_k1_KO_s004 <= 1
 0 <= w_k1_KO_s005 <= 1
 0 <= w_k1_KO_s006 <= 1
 0 <= w_k1_KO_s007 <= 1
 0 <= w_k1_KO_s010 <= 1
 0 <= w_k1_KO_s011 <= 1
 0 <= w_k1_KO_s012 <= 1
 0 <= w_k1_KO_s013 <= 1
 0 <= w_k1_KO_s014 <= 1
 0 <= w_k1_KO_s015 <= 1
 0 <= w_k1_KO_s016 <= 1
 0 <= w_k1_KO_s017 <= 1
 0 <= w_k1_KO_s100 <= 1
 0 <= w_k1_KO_s101 <= 1
 0 <= w_k1_KO_s102 <= 1
 0 <= w_k1_KO_s103 <= 1
 0 <= w_k1_KO_s104 <= 1
 0 <= w_k1_KO_s105 <= 1
 0 <= w_k1_KO_s106 <= 1
 0 <= w_k1_KO_s107 <= 1
 0 <= w_k1_KO_s110 <= 1
 0 <= w_k1_KO_s111 <= 1
 0 <= w_k1_KO_s112 <= 1
 0 <= w_k1_KO_s113 <= 1
 0 <= w_k1_KO_s114 <= 1
 0 <= w_k1_KO_s115 <= 1
 0 <= w_k1_KO_s116 <= 1
 0 <= w_k1_KO_s117 <= 1
 0 <= w_k1_MAT_s0 <= 1
 0 <= w_k1_MAT_s1 <= 1
 0 <= w_k1_MAT_s00 <= 1
 0 <= w_k1_MAT_s01 <= 1
 0 <= w_k1_MAT_s10 <= 1
 0 <= w_k1_MAT_s11 <= 1
 0 <= w_k1_MAT_s000 <= 1
 0 <= w_k1_MAT_s001 <= 1
 0 <= w_k1_MAT_s002 <= 1
 0 <= w_k1_MAT_s003 <= 1
 0 <= w_k1_MAT_s004 <= 1
 0 <= w_k1_MAT_s005 <= 1
 0 <= w_k1_MAT_s006 <= 1
 0 <= w_k1_MAT_s007 <= 1
 0 <= w_k1_MAT_s010 <= 1
 0 <= w_k1_MAT_s011 <= 1
 0 <= w_k1_MAT_s012 <= 1
 0 <= w_k1_MAT_s013 <= 1
 0 <= w_k1_MAT_s014 <= 1
 0 <= w_k1_MAT_s015 <= 1
 0 <= w_k1_MAT_s016 <= 1
 0 <= w_k1_MAT_s017 <= 1
 0 <= w_k1_MAT_s100 <= 1
 0 <= w_k1_MAT_s101 <= 1
 0 <= w_k1_MAT_s102 <= 1
 0 <= w_k1_MAT_s103 <= 1
 0 <= w_k1_MAT_s104 <= 1
 0 <= w_k1_MAT_s105 <= 1
 0 <= w_k1_MAT_s106 <= 1
 0 <= w_k1_MAT_s107 <= 1
 0 <= w_k1_MAT_s110 <= 1
 0 <= w_k1_MAT_s111 <= 1
 0 <= w_k1_MAT_s112 <= 1
 0 <= w_k1_MAT_s113 <= 1
 0 <= w_k1_MAT_s114 <= 1
 0 <= w_k1_MAT_s115 <= 1
 0 <= w_k1_MAT_s116 <= 1
 0 <= w_k1_MAT_s117 <= 1
 0 <= w_k1_MSFT_s0 <= 1
 0 <= w_k1_MSFT_s1 <= 1
 0 <= w_k1_MSFT_s00 <= 1
 0 <= w_k1_MSFT_s01 <= 1
 0 <= w_k1_MSFT_s10 <= 1
 0 <= w_k1_MSFT_s11 <= 1
 0 <= w_k1_MSFT_s000 <= 1
 0 <= w_k1_MSFT_s001 <= 1
 0 <= w_k1_MSFT_s002 <= 1
 0 <= w_k1_MSFT_s003 <= 1
 0 <= w_k1_MSFT_s004 <= 1
 0 <= w_k1_MSFT_s005 <= 1
 0 <= w_k1_MSFT_s006 <= 1
 0 <= w_k1_MSFT_s007 <= 1
 0 <= w_k1_MSFT_s010 <= 1
 0 <= w_k1_MSFT_s011 <= 1
 0 <= w_k1_MSFT_s012 <= 1
 0 <= w_k1_MSFT_s013 <= 1
 0 <= w_k1_MSFT_s014 <= 1
 0 <= w_k1_MSFT_s015 <= 1
 0 <= w_k1_MSFT_s016 <= 1
 0 <= w_k1_MSFT_s017 <= 1
 0 <= w_k1_MSFT_s100 <= 1
 0 <= w_k1_MSFT_s101 <= 1
 0 <= w_k1_MSFT_s102 <= 1
 0 <= w_k1_MSFT_s103 <= 1
 0 <= w_k1_MSFT_s104 <= 1
 0 <= w_k1_MSFT_s105 <= 1
 0 <= w_k1_MSFT_s106 <= 1
 0 <= w_k1_MSFT_s107 <= 1
 0 <= w_k1_MSFT_s110 <= 1
 0 <= w_k1_MSFT_s111 <= 1
 0 <= w_k1_MSFT_s112 <= 1
 0 <= w_k1_MSFT_s113 <= 1
 0 <= w_k1_MSFT_s114 <= 1
 0 <= w_k1_MSFT_s115 <= 1
 0 <= w_k1_MSFT_s116 <= 1
 0 <= w_k1_MSFT_s117 <= 1
 0 <= w_k1_PG_s0 <= 1
 0 <= w_k1_PG_s1 <= 1
 0 <= w_k1_PG_s00 <= 1
 0 <= w_k1_PG_s01 <= 1
 0 <= w_k1_PG_s10 <= 1
 0 <= w_k1_PG_s11 <= 1
 0 <= w_k1_PG_s000 <= 1
 0 <= w_k1_PG_s001 <= 1
 0 <= w_k1_PG_s002 <= 1
 0 <= w_k1_PG_s003 <= 1
 0 <= w_k1_PG_s004 <= 1
 0 <= w_k1_PG_s005 <= 1
 0 <= w_k1_PG_s006 <= 1
 0 <= w_k1_PG_s007 <= 1
 0 <= w_k1_PG_s010 <= 1
 0 <= w_k1_PG_s011 <= 1
 0 <= w_k1_PG_s012 <= 1
 0 <= w_k1_PG_s013 <= 1
 0 <= w_k1_PG_s014 <= 1
 0 <= w_k1_PG_s015 <= 1
 0 <= w_k1_PG_s016 <= 1
 0 <= w_k1_PG_s017 <= 1
 0 <= w_k1_PG_s100 <= 1
 0 <= w_k1_PG_s101 <= 1
 0 <= w_k1_PG_s102 <= 1
 0 <= w_k1_PG_s103 <= 1
 0 <= w_k1_PG_s104 <= 1
 0 <= w_k1_PG_s105 <= 1
 0 <= w_k1_PG_s106 <= 1
 0 <= w_k1_PG_s107 <= 1
 0 <= w_k1_PG_s110 <= 1
 0 <= w_k1_PG_s111 <= 1
 0 <= w_k1_PG_s112 <= 1
 0 <= w_k1_PG_s113 <= 1
 0 <= w_k1_PG_s114 <= 1
 0 <= w_k1_PG_s115 <= 1
 0 <= w_k1_PG_s116 <= 1
 0 <= w_k1_PG_s117 <= 1
 0 <= b_0_ADM <= 0.2
 0 <= b_0_AXP <= 0.2
 0 <= b_0_DIS <= 0.2
 0 <= b_0_FDX <= 0.2
 0 <= b_0_IBM <= 0.2
 0 <= b_0_INTC <= 0.2
 0 <= b_0_KO <= 0.2
 0 <= b_0_MAT <= 0.2
 0 <= b_0_MSFT <= 0.2
 0 <= b_0_PG <= 0.2
 0 <= b_k1_ADM_s0 <= 0.2
 0 <= b_k1_ADM_s1 <= 0.2
 0 <= b_k1_ADM_s00 <= 0.2
 0 <= b_k1_ADM_s01 <= 0.2
 0 <= b_k1_ADM_s10 <= 0.2
 0 <= b_k1_ADM_s11 <= 0.2
 0 <= b_k1_ADM_s000 <= 0.2
 0 <= b_k1_ADM_s001 <= 0.2
 0 <= b_k1_ADM_s002 <= 0.2
 0 <= b_k1_ADM_s003 <= 0.2
 0 <= b_k1_ADM_s004 <= 0.2
 0 <= b_k1_ADM_s005 <= 0.2
 0 <= b_k1_ADM_s006 <= 0.2
 0 <= b_k1_ADM_s007 <= 0.2
 0 <= b_k1_ADM_s010 <= 0.2
 0 <= b_k1_ADM_s011 <= 0.2
 0 <= b_k1_ADM_s012 <= 0.2
 0 <= b_k1_ADM_s013 <= 0.2
 0 <= b_k1_ADM_s014 <= 0.2
 0 <= b_k1_ADM_s015 <= 0.2
 0 <= b_k1_ADM_s016 <= 0.2
 0 <= b_k1_ADM_s017 <= 0.2
 0 <= b_k1_ADM_s100 <= 0.2
 0 <= b_k1_ADM_s101 <= 0.2
 0 <= b_k1_ADM_s102 <= 0.2
 0 <= b_k1_ADM_s103 <= 0.2
 0 <= b_k1_ADM_s104 <= 0.2
 0 <= b_k1_ADM_s105 <= 0.2
 0 <= b_k1_ADM_s106 <= 0.2
 0 <= b_k1_ADM_s107 <= 0.2
 0 <= b_k1_ADM_s110 <= 0.2
 0 <= b_k1_ADM_s111 <= 0.2
 0 <= b_k1_ADM_s112 <= 0.2
 0 <= b_k1_ADM_s113 <= 0.2
 0 <= b_k1_ADM_s114 <= 0.2
 0 <= b_k1_ADM_s115 <= 0.2
 0 <= b_k1_ADM_s116 <= 0.2
 0 <= b_k1_ADM_s117 <= 0.2
 0 <= b_k1_AXP_s0 <= 0.2
 0 <= b_k1_AXP_s1 <= 0.2
 0 <= b_k1_AXP_s00 <= 0.2
 0 <= b_k1_AXP_s01 <= 0.2
 0 <= b_k1_AXP_s10 <= 0.2
 0 <= b_k1_AXP_s11 <= 0.2
 0 <= b_k1_AXP_s000 <= 0.2
 0 <= b_k1_AXP_s001 <= 0.2
 0 <= b_k1_AXP_s002 <= 0.2
 0 <= b_k1_AXP_s003 <= 0.2
 0 <= b_k1_AXP_s004 <= 0.2
 0 <= b_k1_AXP_s005 <= 0.2
 0 <= b_k1_AXP_s006 <= 0.2
 0 <= b_k1_AXP_s007 <= 0.2
 0 <= b_k1_AXP_s010 <= 0.2
 0 <= b_k1_AXP_s011 <= 0.2
 0 <= b_k1_AXP_s012 <= 0.2
 0 <= b_k1_AXP_s013 <= 0.2
 0 <= b_k1_AXP_s014 <= 0.2
 0 <= b_k1_AXP_s015 <= 0.2
 0 <= b_k1_AXP_s016 <= 0.2
 0 <= b_k1_AXP_s017 <= 0.2
 0 <= b_k1_AXP_s100 <= 0.2
 0 <= b_k1_AXP_s101 <= 0.2
 0 <= b_k1_AXP_s102 <= 0.2
 0 <= b_k1_AXP_s103 <= 0.2
 0 <= b_k1_AXP_s104 <= 0.2
 0 <= b_k1_AXP_s105 <= 0.2
 0 <= b_k1_AXP_s106 <= 0.2
 0 <= b_k1_AXP_s107 <= 0.2
 0 <= b_k1_AXP_s110 <= 0.2
 0 <= b_k1_AXP_s111 <= 0.2
 0 <= b_k1_AXP_s112 <= 0.2
 0 <= b_k1_AXP_s113 <= 0.2
 0 <= b_k1_AXP_s114 <= 0.2
 0 <= b_k1_AXP_s115 <= 0.2
 0 <= b_k1_AXP_s116 <= 0.2
 0 <= b_k1_AXP_s117 <= 0.2
 0 <= b_k1_DIS_s0 <= 0.2
 0 <= b_k1_DIS_s1 <= 0.2
 0 <= b_k1_DIS_s00 <= 0.2
 0 <= b_k1_DIS_s01 <= 0.2
 0 <= b_k1_DIS_s10 <= 0.2
 0 <= b_k1_DIS_s11 <= 0.2
 0 <= b_k1_DIS_s000 <= 0.2
 0 <= b_k1_DIS_s001 <= 0.2
 0 <= b_k1_DIS_s002 <= 0.2
 0 <= b_k1_DIS_s003 <= 0.2
 0 <= b_k1_DIS_s004 <= 0.2
 0 <= b_k1_DIS_s005 <= 0.2
 0 <= b_k1_DIS_s006 <= 0.2
 0 <= b_k1_DIS_s007 <= 0.2
 0 <= b_k1_DIS_s010 <= 0.2
 0 <= b_k1_DIS_s011 <= 0.2
 0 <= b_k1_DIS_s012 <= 0.2
 0 <= b_k1_DIS_s013 <= 0.2
 0 <= b_k1_DIS_s014 <= 0.2
 0 <= b_k1_DIS_s015 <= 0.2
 0 <= b_k1_DIS_s016 <= 0.2
 0 <= b_k1_DIS_s017 <= 0.2
 0 <= b_k1_DIS_s100 <= 0.2
 0 <= b_k1_DIS_s101 <= 0.2
 0 <= b_k1_DIS_s102 <= 0.2
 0 <= b_k1_DIS_s103 <= 0.2
 0 <= b_k1_DIS_s104 <= 0.2
 0 <= b_k1_DIS_s105 <= 0.2
 0 <= b_k1_DIS_s106 <= 0.2
 0 <= b_k1_DIS_s107 <= 0.2
 0 <= b_k1_DIS_s110 <= 0.2
 0 <= b_k1_DIS_s111 <= 0.2
 0 <= b_k1_DIS_s112 <= 0.2
 0 <= b_k1_DIS_s113 <= 0.2
 0 <= b_k1_DIS_s114 <= 0.2
 0 <= b_k1_DIS_s115 <= 0.2
 0 <= b_k1_DIS_s116 <= 0.2
 0 <= b_k1_DIS_s117 <= 0.2
 0 <= b_k1_FDX_s0 <= 0.2
 0 <= b_k1_FDX_s1 <= 0.2
 0 <= b_k1_FDX_s00 <= 0.2
 0 <= b_k1_FDX_s01 <= 0.2
 0 <= b_k1_FDX_s10 <= 0.2
 0 <= b_k1_FDX_s11 <= 0.2
 0 <= b_k1_FDX_s000 <= 0.2
 0 <= b_k1_FDX_s001 <= 0.2
 0 <= b_k1_FDX_s002 <= 0.2
 0 <= b_k1_FDX_s003 <= 0.2
 0 <= b_k1_FDX_s004 <= 0.2
 0 <= b_k1_FDX_s005 <= 0.2
 0 <= b_k1_FDX_s006 <= 0.2
 0 <= b_k1_FDX_s007 <= 0.2
 0 <= b_k1_FDX_s010 <= 0.2
 0 <= b_k1_FDX_s011 <= 0.2
 0 <= b_k1_FDX_s012 <= 0.2
 0 <= b_k1_FDX_s013 <= 0.2
 0 <= b_k1_FDX_s014 <= 0.2
 0 <= b_k1_FDX_s015 <= 0.2
 0 <= b_k1_FDX_s016 <= 0.2
 0 <= b_k1_FDX_s017 <= 0.2
 0 <= b_k1_FDX_s100 <= 0.2
 0 <= b_k1_FDX_s101 <= 0.2
 0 <= b_k1_FDX_s102 <= 0.2
 0 <= b_k1_FDX_s103 <= 0.2
 0 <= b_k1_FDX_s104 <= 0.2
 0 <= b_k1_FDX_s105 <= 0.2
 0 <= b_k1_FDX_s106 <= 0.2
 0 <= b_k1_FDX_s107 <= 0.2
 0 <= b_k1_FDX_s110 <= 0.2
 0 <= b_k1_FDX_s111 <= 0.2
 0 <= b_k1_FDX_s112 <= 0.2
 0 <= b_k1_FDX_s113 <= 0.2
 0 <= b_k1_FDX_s114 <= 0.2
 0 <= b_k1_FDX_s115 <= 0.2
 0 <= b_k1_FDX_s116 <= 0.2
 0 <= b_k1_FDX_s117 <= 0.2
 0 <= b_k1_IBM_s0 <= 0.2
 0 <= b_k1_IBM_s1 <= 0.2
 0 <= b_k1_IBM_s00 <= 0.2
 0 <= b_k1_IBM_s01 <= 0.2
 0 <= b_k1_IBM_s10 <= 0.2
 0 <= b_k1_IBM_s11 <= 0.2
 0 <= b_k1_IBM_s000 <= 0.2
 0 <= b_k1_IBM_s001 <= 0.2
 0 <= b_k1_IBM_s002 <= 0.2
 0 <= b_k1_IBM_s003 <= 0.2
 0 <= b_k1_IBM_s004 <= 0.2
 0 <= b_k1_IBM_s005 <= 0.2
 0 <= b_k1_IBM_s006 <= 0.2
 0 <= b_k1_IBM_s007 <= 0.2
 0 <= b_k1_IBM_s010 <= 0.2
 0 <= b_k1_IBM_s011 <= 0.2
 0 <= b_k1_IBM_s012 <= 0.2
 0 <= b_k1_IBM_s013 <= 0.2
 0 <= b_k1_IBM_s014 <= 0.2
 0 <= b_k1_IBM_s015 <= 0.2
 0 <= b_k1_IBM_s016 <= 0.2
 0 <= b_k1_IBM_s017 <= 0.2
 0 <= b_k1_IBM_s100 <= 0.2
 0 <= b_k1_IBM_s101 <= 0.2
 0 <= b_k1_IBM_s102 <= 0.2
 0 <= b_k1_IBM_s103 <= 0.2
 0 <= b_k1_IBM_s104 <= 0.2
 0 <= b_k1_IBM_s105 <= 0.2
 0 <= b_k1_IBM_s106 <= 0.2
 0 <= b_k1_IBM_s107 <= 0.2
 0 <= b_k1_IBM_s110 <= 0.2
 0 <= b_k1_IBM_s111 <= 0.2
 0 <= b_k1_IBM_s112 <= 0.2
 0 <= b_k1_IBM_s113 <= 0.2
 0 <= b_k1_IBM_s114 <= 0.2
 0 <= b_k1_IBM_s115 <= 0.2
 0 <= b_k1_IBM_s116 <= 0.2
 0 <= b_k1_IBM_s117 <= 0.2
 0 <= b_k1_INTC_s0 <= 0.2
 0 <= b_k1_INTC_s1 <= 0.2
 0 <= b_k1_INTC_s00 <= 0.2
 0 <= b_k1_INTC_s01 <= 0.2
 0 <= b_k1_INTC_s10 <= 0.2
 0 <= b_k1_INTC_s11 <= 0.2
 0 <= b_k1_INTC_s000 <= 0.2
 0 <= b_k1_INTC_s001 <= 0.2
 0 <= b_k1_INTC_s002 <= 0.2
 0 <= b_k1_INTC_s003 <= 0.2
 0 <= b_k1_INTC_s004 <= 0.2
 0 <= b_k1_INTC_s005 <= 0.2
 0 <= b_k1_INTC_s006 <= 0.2
 0 <= b_k1_INTC_s007 <= 0.2
 0 <= b_k1_INTC_s010 <= 0.2
 0 <= b_k1_INTC_s011 <= 0.2
 0 <= b_k1_INTC_s012 <= 0.2
 0 <= b_k1_INTC_s013 <= 0.2
 0 <= b_k1_INTC_s014 <= 0.2
 0 <= b_k1_INTC_s015 <= 0.2
 0 <= b_k1_INTC_s016 <= 0.2
 0 <= b_k1_INTC_s017 <= 0.2
 0 <= b_k1_INTC_s100 <= 0.2
 0 <= b_k1_INTC_s101 <= 0.2
 0 <= b_k1_INTC_s102 <= 0.2
 0 <= b_k1_INTC_s103 <= 0.2
 0 <= b_k1_INTC_s104 <= 0.2
 0 <= b_k1_INTC_s105 <= 0.2
 0 <= b_k1_INTC_s106 <= 0.2
 0 <= b_k1_INTC_s107 <= 0.2
 0 <= b_k1_INTC_s110 <= 0.2
 0 <= b_k1_INTC_s111 <= 0.2
 0 <= b_k1_INTC_s112 <= 0.2
 0 <= b_k1_INTC_s113 <= 0.2
 0 <= b_k1_INTC_s114 <= 0.2
 0 <= b_k1_INTC_s115 <= 0.2
 0 <= b_k1_INTC_s116 <= 0.2
 0 <= b_k1_INTC_s117 <= 0.2
 0 <= b_k1_KO_s0 <= 0.2
 0 <= b_k1_KO_s1 <= 0.2
 0 <= b_k1_KO_s00 <= 0.2
 0 <= b_k1_KO_s01 <= 0.2
 0 <= b_k1_KO_s10 <= 0.2
 0 <= b_k1_KO_s11 <= 0.2
 0 <= b_k1_KO_s000 <= 0.2
 0 <= b_k1_KO_s001 <= 0.2
 0 <= b_k1_KO_s002 <= 0.2
 0 <= b_k1_KO_s003 <= 0.2
 0 <= b_k1_KO_s004 <= 0.2
 0 <= b_k1_KO_s005 <= 0.2
 0 <= b_k1_KO_s006 <= 0.2
 0 <= b_k1_KO_s007 <= 0.2
 0 <= b_k1_KO_s010 <= 0.2
 0 <= b_k1_KO_s011 <= 0.2
 0 <= b_k1_KO_s012 <= 0.2
 0 <= b_k1_KO_s013 <= 0.2
 0 <= b_k1_KO_s014 <= 0.2
 0 <= b_k1_KO_s015 <= 0.2
 0 <= b_k1_KO_s016 <= 0.2
 0 <= b_k1_KO_s017 <= 0.2
 0 <= b_k1_KO_s100 <= 0.2
 0 <= b_k1_KO_s101 <= 0.2
 0 <= b_k1_KO_s102 <= 0.2
 0 <= b_k1_KO_s103 <= 0.2
 0 <= b_k1_KO_s104 <= 0.2
 0 <= b_k1_KO_s105 <= 0.2
 0 <= b_k1_KO_s106 <= 0.2
 0 <= b_k1_KO_s107 <= 0.2
 0 <= b_k1_KO_s110 <= 0.2
 0 <= b_k1_KO_s111 <= 0.2
 0 <= b_k1_KO_s112 <= 0.2
 0 <= b_k1_KO_s113 <= 0.2
 0 <= b_k1_KO_s114 <= 0.2
 0 <= b_k1_KO_s115 <= 0.2
 0 <= b_k1_KO_s116 <= 0.2
 0 <= b_k1_KO_s117 <= 0.2
 0 <= b_k1_MAT_s0 <= 0.2
 0 <= b_k1_MAT_s1 <= 0.2
 0 <= b_k1_MAT_s00 <= 0.2
 0 <= b_k1_MAT_s01 <= 0.2
 0 <= b_k1_MAT_s10 <= 0.2
 0 <= b_k1_MAT_s11 <= 0.2
 0 <= b_k1_MAT_s000 <= 0.2
 0 <= b_k1_MAT_s001 <= 0.2
 0 <= b_k1_MAT_s002 <= 0.2
 0 <= b_k1_MAT_s003 <= 0.2
 0 <= b_k1_MAT_s004 <= 0.2
 0 <= b_k1_MAT_s005 <= 0.2
 0 <= b_k1_MAT_s006 <= 0.2
 0 <= b_k1_MAT_s007 <= 0.2
 0 <= b_k1_MAT_s010 <= 0.2
 0 <= b_k1_MAT_s011 <= 0.2
 0 <= b_k1_MAT_s012 <= 0.2
 0 <= b_k1_MAT_s013 <= 0.2
 0 <= b_k1_MAT_s014 <= 0.2
 0 <= b_k1_MAT_s015 <= 0.2
 0 <= b_k1_MAT_s016 <= 0.2
 0 <= b_k1_MAT_s017 <= 0.2
 0 <= b_k1_MAT_s100 <= 0.2
 0 <= b_k1_MAT_s101 <= 0.2
 0 <= b_k1_MAT_s102 <= 0.2
 0 <= b_k1_MAT_s103 <= 0.2
 0 <= b_k1_MAT_s104 <= 0.2
 0 <= b_k1_MAT_s105 <= 0.2
 0 <= b_k1_MAT_s106 <= 0.2
 0 <= b_k1_MAT_s107 <= 0.2
 0 <= b_k1_MAT_s110 <= 0.2
 0 <= b_k1_MAT_s111 <= 0.2
 0 <= b_k1_MAT_s112 <= 0.2
 0 <= b_k1_MAT_s113 <= 0.2
 0 <= b_k1_MAT_s114 <= 0.2
 0 <= b_k1_MAT_s115 <= 0.2
 0 <= b_k1_MAT_s116 <= 0.2
 0 <= b_k1_MAT_s117 <= 0.2
 0 <= b_k1_MSFT_s0 <= 0.2
 0 <= b_k1_MSFT_s1 <= 0.2
 0 <= b_k1_MSFT_s00 <= 0.2
 0 <= b_k1_MSFT_s01 <= 0.2
 0 <= b_k1_MSFT_s10 <= 0.2
 0 <= b_k1_MSFT_s11 <= 0.2
 0 <= b_k1_MSFT_s000 <= 0.2
 0 <= b_k1_MSFT_s001 <= 0.2
 0 <= b_k1_MSFT_s002 <= 0.2
 0 <= b_k1_MSFT_s003 <= 0.2
 0 <= b_k1_MSFT_s004 <= 0.2
 0 <= b_k1_MSFT_s005 <= 0.2
 0 <= b_k1_MSFT_s006 <= 0.2
 0 <= b_k1_MSFT_s007 <= 0.2
 0 <= b_k1_MSFT_s010 <= 0.2
 0 <= b_k1_MSFT_s011 <= 0.2
 0 <= b_k1_MSFT_s012 <= 0.2
 0 <= b_k1_MSFT_s013 <= 0.2
 0 <= b_k1_MSFT_s014 <= 0.2
 0 <= b_k1_MSFT_s015 <= 0.2
 0 <= b_k1_MSFT_s016 <= 0.2
 0 <= b_k1_MSFT_s017 <= 0.2
 0 <= b_k1_MSFT_s100 <= 0.2
 0 <= b_k1_MSFT_s101 <= 0.2
 0 <= b_k1_MSFT_s102 <= 0.2
 0 <= b_k1_MSFT_s103 <= 0.2
 0 <= b_k1_MSFT_s104 <= 0.2
 0 <= b_k1_MSFT_s105 <= 0.2
 0 <= b_k1_MSFT_s106 <= 0.2
 0 <= b_k1_MSFT_s107 <= 0.2
 0 <= b_k1_MSFT_s110 <= 0.2
 0 <= b_k1_MSFT_s111 <= 0.2
 0 <= b_k1_MSFT_s112 <= 0.2
 0 <= b_k1_MSFT_s113 <= 0.2
 0 <= b_k1_MSFT_s114 <= 0.2
 0 <= b_k1_MSFT_s115 <= 0.2
 0 <= b_k1_MSFT_s116 <= 0.2
 0 <= b_k1_MSFT_s117 <= 0.2
 0 <= b_k1_PG_s0 <= 0.2
 0 <= b_k1_PG_s1 <= 0.2
 0 <= b_k1_PG_s00 <= 0.2
 0 <= b_k1_PG_s01 <= 0.2
 0 <= b_k1_PG_s10 <= 0.2
 0 <= b_k1_PG_s11 <= 0.2
 0 <= b_k1_PG_s000 <= 0.2
 0 <= b_k1_PG_s001 <= 0.2
 0 <= b_k1_PG_s002 <= 0.2
 0 <= b_k1_PG_s003 <= 0.2
 0 <= b_k1_PG_s004 <= 0.2
 0 <= b_k1_PG_s005 <= 0.2
 0 <= b_k1_PG_s006 <= 0.2
 0 <= b_k1_PG_s007 <= 0.2
 0 <= b_k1_PG_s010 <= 0.2
 0 <= b_k1_PG_s011 <= 0.2
 0 <= b_k1_PG_s012 <= 0.2
 0 <= b_k1_PG_s013 <= 0.2
 0 <= b_k1_PG_s014 <= 0.2
 0 <= b_k1_PG_s015 <= 0.2
 0 <= b_k1_PG_s016 <= 0.2
 0 <= b_k1_PG_s017 <= 0.2
 0 <= b_k1_PG_s100 <= 0.2
 0 <= b_k1_PG_s101 <= 0.2
 0 <= b_k1_PG_s102 <= 0.2
 0 <= b_k1_PG_s103 <= 0.2
 0 <= b_k1_PG_s104 <= 0.2
 0 <= b_k1_PG_s105 <= 0.2
 0 <= b_k1_PG_s106 <= 0.2
 0 <= b_k1_PG_s107 <= 0.2
 0 <= b_k1_PG_s110 <= 0.2
 0 <= b_k1_PG_s111 <= 0.2
 0 <= b_k1_PG_s112 <= 0.2
 0 <= b_k1_PG_s113 <= 0.2
 0 <= b_k1_PG_s114 <= 0.2
 0 <= b_k1_PG_s115 <= 0.2
 0 <= b_k1_PG_s116 <= 0.2
 0 <= b_k1_PG_s117 <= 0.2
 0 <= s_0_ADM <= 0.2
 0 <= s_0_AXP <= 0.2
 0 <= s_0_DIS <= 0.2
 0 <= s_0_FDX <= 0.2
 0 <= s_0_IBM <= 0.2
 0 <= s_0_INTC <= 0.2
 0 <= s_0_KO <= 0.2
 0 <= s_0_MAT <= 0.2
 0 <= s_0_MSFT <= 0.2
 0 <= s_0_PG <= 0.2
 0 <= s_k1_ADM_s0 <= 0.2
 0 <= s_k1_ADM_s1 <= 0.2
 0 <= s_k1_ADM_s00 <= 0.2
 0 <= s_k1_ADM_s01 <= 0.2
 0 <= s_k1_ADM_s10 <= 0.2
 0 <= s_k1_ADM_s11 <= 0.2
 0 <= s_k1_ADM_s000 <= 0.2
 0 <= s_k1_ADM_s001 <= 0.2
 0 <= s_k1_ADM_s002 <= 0.2
 0 <= s_k1_ADM_s003 <= 0.2
 0 <= s_k1_ADM_s004 <= 0.2
 0 <= s_k1_ADM_s005 <= 0.2
 0 <= s_k1_ADM_s006 <= 0.2
 0 <= s_k1_ADM_s007 <= 0.2
 0 <= s_k1_ADM_s010 <= 0.2
 0 <= s_k1_ADM_s011 <= 0.2
 0 <= s_k1_ADM_s012 <= 0.2
 0 <= s_k1_ADM_s013 <= 0.2
 0 <= s_k1_ADM_s014 <= 0.2
 0 <= s_k1_ADM_s015 <= 0.2
 0 <= s_k1_ADM_s016 <= 0.2
 0 <= s_k1_ADM_s017 <= 0.2
 0 <= s_k1_ADM_s100 <= 0.2
 0 <= s_k1_ADM_s101 <= 0.2
 0 <= s_k1_ADM_s102 <= 0.2
 0 <= s_k1_ADM_s103 <= 0.2
 0 <= s_k1_ADM_s104 <= 0.2
 0 <= s_k1_ADM_s105 <= 0.2
 0 <= s_k1_ADM_s106 <= 0.2
 0 <= s_k1_ADM_s107 <= 0.2
 0 <= s_k1_ADM_s110 <= 0.2
 0 <= s_k1_ADM_s111 <= 0.2
 0 <= s_k1_ADM_s112 <= 0.2
 0 <= s_k1_ADM_s113 <= 0.2
 0 <= s_k1_ADM_s114 <= 0.2
 0 <= s_k1_ADM_s115 <= 0.2
 0 <= s_k1_ADM_s116 <= 0.2
 0 <= s_k1_ADM_s117 <= 0.2
 0 <= s_k1_AXP_s0 <= 0.2
 0 <= s_k1_AXP_s1 <= 0.2
 0 <= s_k1_AXP_s00 <= 0.2
 0 <= s_k1_AXP_s01 <= 0.2
 0 <= s_k1_AXP_s10 <= 0.2
 0 <= s_k1_AXP_s11 <= 0.2
 0 <= s_k1_AXP_s000 <= 0.2
 0 <= s_k1_AXP_s001 <= 0.2
 0 <= s_k1_AXP_s002 <= 0.2
 0 <= s_k1_AXP_s003 <= 0.2
 0 <= s_k1_AXP_s004 <= 0.2
 0 <= s_k1_AXP_s005 <= 0.2
 0 <= s_k1_AXP_s006 <= 0.2
 0 <= s_k1_AXP_s007 <= 0.2
 0 <= s_k1_AXP_s010 <= 0.2
 0 <= s_k1_AXP_s011 <= 0.2
 0 <= s_k1_AXP_s012 <= 0.2
 0 <= s_k1_AXP_s013 <= 0.2
 0 <= s_k1_AXP_s014 <= 0.2
 0 <= s_k1_AXP_s015 <= 0.2
 0 <= s_k1_AXP_s016 <= 0.2
 0 <= s_k1_AXP_s017 <= 0.2
 0 <= s_k1_AXP_s100 <= 0.2
 0 <= s_k1_AXP_s101 <= 0.2
 0 <= s_k1_AXP_s102 <= 0.2
 0 <= s_k1_AXP_s103 <= 0.2
 0 <= s_k1_AXP_s104 <= 0.2
 0 <= s_k1_AXP_s105 <= 0.2
 0 <= s_k1_AXP_s106 <= 0.2
 0 <= s_k1_AXP_s107 <= 0.2
 0 <= s_k1_AXP_s110 <= 0.2
 0 <= s_k1_AXP_s111 <= 0.2
 0 <= s_k1_AXP_s112 <= 0.2
 0 <= s_k1_AXP_s113 <= 0.2
 0 <= s_k1_AXP_s114 <= 0.2
 0 <= s_k1_AXP_s115 <= 0.2
 0 <= s_k1_AXP_s116 <= 0.2
 0 <= s_k1_AXP_s117 <= 0.2
 0 <= s_k1_DIS_s0 <= 0.2
 0 <= s_k1_DIS_s1 <= 0.2
 0 <= s_k1_DIS_s00 <= 0.2
 0 <= s_k1_DIS_s01 <= 0.2
 0 <= s_k1_DIS_s10 <= 0.2
 0 <= s_k1_DIS_s11 <= 0.2
 0 <= s_k1_DIS_s000 <= 0.2
 0 <= s_k1_DIS_s001 <= 0.2
 0 <= s_k1_DIS_s002 <= 0.2
 0 <= s_k1_DIS_s003 <= 0.2
 0 <= s_k1_DIS_s004 <= 0.2
 0 <= s_k1_DIS_s005 <= 0.2
 0 <= s_k1_DIS_s006 <= 0.2
 0 <= s_k1_DIS_s007 <= 0.2
 0 <= s_k1_DIS_s010 <= 0.2
 0 <= s_k1_DIS_s011 <= 0.2
 0 <= s_k1_DIS_s012 <= 0.2
 0 <= s_k1_DIS_s013 <= 0.2
 0 <= s_k1_DIS_s014 <= 0.2
 0 <= s_k1_DIS_s015 <= 0.2
 0 <= s_k1_DIS_s016 <= 0.2
 0 <= s_k1_DIS_s017 <= 0.2
 0 <= s_k1_DIS_s100 <= 0.2
 0 <= s_k1_DIS_s101 <= 0.2
 0 <= s_k1_DIS_s102 <= 0.2
 0 <= s_k1_DIS_s103 <= 0.2
 0 <= s_k1_DIS_s104 <= 0.2
 0 <= s_k1_DIS_s105 <= 0.2
 0 <= s_k1_DIS_s106 <= 0.2
 0 <= s_k1_DIS_s107 <= 0.2
 0 <= s_k1_DIS_s110 <= 0.2
 0 <= s_k1_DIS_s111 <= 0.2
 0 <= s_k1_DIS_s112 <= 0.2
 0 <= s_k1_DIS_s113 <= 0.2
 0 <= s_k1_DIS_s114 <= 0.2
 0 <= s_k1_DIS_s115 <= 0.2
 0 <= s_k1_DIS_s116 <= 0.2
 0 <= s_k1_DIS_s117 <= 0.2
 0 <= s_k1_FDX_s0 <= 0.2
 0 <= s_k1_FDX_s1 <= 0.2
 0 <= s_k1_FDX_s00 <= 0.2
 0 <= s_k1_FDX_s01 <= 0.2
 0 <= s_k1_FDX_s10 <= 0.2
 0 <= s_k1_FDX_s11 <= 0.2
 0 <= s_k1_FDX_s000 <= 0.2
 0 <= s_k1_FDX_s001 <= 0.2
 0 <= s_k1_FDX_s002 <= 0.2
 0 <= s_k1_FDX_s003 <= 0.2
 0 <= s_k1_FDX_s004 <= 0.2
 0 <= s_k1_FDX_s005 <= 0.2
 0 <= s_k1_FDX_s006 <= 0.2
 0 <= s_k1_FDX_s007 <= 0.2
 0 <= s_k1_FDX_s010 <= 0.2
 0 <= s_k1_FDX_s011 <= 0.2
 0 <= s_k1_FDX_s012 <= 0.2
 0 <= s_k1_FDX_s013 <= 0.2
 0 <= s_k1_FDX_s014 <= 0.2
 0 <= s_k1_FDX_s015 <= 0.2
 0 <= s_k1_FDX_s016 <= 0.2
 0 <= s_k1_FDX_s017 <= 0.2
 0 <= s_k1_FDX_s100 <= 0.2
 0 <= s_k1_FDX_s101 <= 0.2
 0 <= s_k1_FDX_s102 <= 0.2
 0 <= s_k1_FDX_s103 <= 0.2
 0 <= s_k1_FDX_s104 <= 0.2
 0 <= s_k1_FDX_s105 <= 0.2
 0 <= s_k1_FDX_s106 <= 0.2
 0 <= s_k1_FDX_s107 <= 0.2
 0 <= s_k1_FDX_s110 <= 0.2
 0 <= s_k1_FDX_s111 <= 0.2
 0 <= s_k1_FDX_s112 <= 0.2
 0 <= s_k1_FDX_s113 <= 0.2
 0 <= s_k1_FDX_s114 <= 0.2
 0 <= s_k1_FDX_s115 <= 0.2
 0 <= s_k1_FDX_s116 <= 0.2
 0 <= s_k1_FDX_s117 <= 0.2
 0 <= s_k1_IBM_s0 <= 0.2
 0 <= s_k1_IBM_s1 <= 0.2
 0 <= s_k1_IBM_s00 <= 0.2
 0 <= s_k1_IBM_s01 <= 0.2
 0 <= s_k1_IBM_s10 <= 0.2
 0 <= s_k1_IBM_s11 <= 0.2
 0 <= s_k1_IBM_s000 <= 0.2
 0 <= s_k1_IBM_s001 <= 0.2
 0 <= s_k1_IBM_s002 <= 0.2
 0 <= s_k1_IBM_s003 <= 0.2
 0 <= s_k1_IBM_s004 <= 0.2
 0 <= s_k1_IBM_s005 <= 0.2
 0 <= s_k1_IBM_s006 <= 0.2
 0 <= s_k1_IBM_s007 <= 0.2
 0 <= s_k1_IBM_s010 <= 0.2
 0 <= s_k1_IBM_s011 <= 0.2
 0 <= s_k1_IBM_s012 <= 0.2
 0 <= s_k1_IBM_s013 <= 0.2
 0 <= s_k1_IBM_s014 <= 0.2
 0 <= s_k1_IBM_s015 <= 0.2
 0 <= s_k1_IBM_s016 <= 0.2
 0 <= s_k1_IBM_s017 <= 0.2
 0 <= s_k1_IBM_s100 <= 0.2
 0 <= s_k1_IBM_s101 <= 0.2
 0 <= s_k1_IBM_s102 <= 0.2
 0 <= s_k1_IBM_s103 <= 0.2
 0 <= s_k1_IBM_s104 <= 0.2
 0 <= s_k1_IBM_s105 <= 0.2
 0 <= s_k1_IBM_s106 <= 0.2
 0 <= s_k1_IBM_s107 <= 0.2
 0 <= s_k1_IBM_s110 <= 0.2
 0 <= s_k1_IBM_s111 <= 0.2
 0 <= s_k1_IBM_s112 <= 0.2
 0 <= s_k1_IBM_s113 <= 0.2
 0 <= s_k1_IBM_s114 <= 0.2
 0 <= s_k1_IBM_s115 <= 0.2
 0 <= s_k1_IBM_s116 <= 0.2
 0 <= s_k1_IBM_s117 <= 0.2
 0 <= s_k1_INTC_s0 <= 0.2
 0 <= s_k1_INTC_s1 <= 0.2
 0 <= s_k1_INTC_s00 <= 0.2
 0 <= s_k1_INTC_s01 <= 0.2
 0 <= s_k1_INTC_s10 <= 0.2
 0 <= s_k1_INTC_s11 <= 0.2
 0 <= s_k1_INTC_s000 <= 0.2
 0 <= s_k1_INTC_s001 <= 0.2
 0 <= s_k1_INTC_s002 <= 0.2
 0 <= s_k1_INTC_s003 <= 0.2
 0 <= s_k1_INTC_s004 <= 0.2
 0 <= s_k1_INTC_s005 <= 0.2
 0 <= s_k1_INTC_s006 <= 0.2
 0 <= s_k1_INTC_s007 <= 0.2
 0 <= s_k1_INTC_s010 <= 0.2
 0 <= s_k1_INTC_s011 <= 0.2
 0 <= s_k1_INTC_s012 <= 0.2
 0 <= s_k1_INTC_s013 <= 0.2
 0 <= s_k1_INTC_s014 <= 0.2
 0 <= s_k1_INTC_s015 <= 0.2
 0 <= s_k1_INTC_s016 <= 0.2
 0 <= s_k1_INTC_s017 <= 0.2
 0 <= s_k1_INTC_s100 <= 0.2
 0 <= s_k1_INTC_s101 <= 0.2
 0 <= s_k1_INTC_s102 <= 0.2
 0 <= s_k1_INTC_s103 <= 0.2
 0 <= s_k1_INTC_s104 <= 0.2
 0 <= s_k1_INTC_s105 <= 0.2
 0 <= s_k1_INTC_s106 <= 0.2
 0 <= s_k1_INTC_s107 <= 0.2
 0 <= s_k1_INTC_s110 <= 0.2
 0 <= s_k1_INTC_s111 <= 0.2
 0 <= s_k1_INTC_s112 <= 0.2
 0 <= s_k1_INTC_s113 <= 0.2
 0 <= s_k1_INTC_s114 <= 0.2
 0 <= s_k1_INTC_s115 <= 0.2
 0 <= s_k1_INTC_s116 <= 0.2
 0 <= s_k1_INTC_s117 <= 0.2
 0 <= s_k1_KO_s0 <= 0.2
 0 <= s_k1_KO_s1 <= 0.2
 0 <= s_k1_KO_s00 <= 0.2
 0 <= s_k1_KO_s01 <= 0.2
 0 <= s_k1_KO_s10 <= 0.2
 0 <= s_k1_KO_s11 <= 0.2
 0 <= s_k1_KO_s000 <= 0.2
 0 <= s_k1_KO_s001 <= 0.2
 0 <= s_k1_KO_s002 <= 0.2
 0 <= s_k1_KO_s003 <= 0.2
 0 <= s_k1_KO_s004 <= 0.2
 0 <= s_k1_KO_s005 <= 0.2
 0 <= s_k1_KO_s006 <= 0.2
 0 <= s_k1_KO_s007 <= 0.2
 0 <= s_k1_KO_s010 <= 0.2
 0 <= s_k1_KO_s011 <= 0.2
 0 <= s_k1_KO_s012 <= 0.2
 0 <= s_k1_KO_s013 <= 0.2
 0 <= s_k1_KO_s014 <= 0.2
 0 <= s_k1_KO_s015 <= 0.2
 0 <= s_k1_KO_s016 <= 0.2
 0 <= s_k1_KO_s017 <= 0.2
 0 <= s_k1_KO_s100 <= 0.2
 0 <= s_k1_KO_s101 <= 0.2
 0 <= s_k1_KO_s102 <= 0.2
 0 <= s_k1_KO_s103 <= 0.2
 0 <= s_k1_KO_s104 <= 0.2
 0 <= s_k1_KO_s105 <= 0.2
 0 <= s_k1_KO_s106 <= 0.2
 0 <= s_k1_KO_s107 <= 0.2
 0 <= s_k1_KO_s110 <= 0.2
 0 <= s_k1_KO_s111 <= 0.2
 0 <= s_k1_KO_s112 <= 0.2
 0 <= s_k1_KO_s113 <= 0.2
 0 <= s_k1_KO_s114 <= 0.2
 0 <= s_k1_KO_s115 <= 0.2
 0 <= s_k1_KO_s116 <= 0.2
 0 <= s_k1_KO_s117 <= 0.2
 0 <= s_k1_MAT_s0 <= 0.2
 0 <= s_k1_MAT_s1 <= 0.2
 0 <= s_k1_MAT_s00 <= 0.2
 0 <= s_k1_MAT_s01 <= 0.2
 0 <= s_k1_MAT_s10 <= 0.2
 0 <= s_k1_MAT_s11 <= 0.2
 0 <= s_k1_MAT_s000 <= 0.2
 0 <= s_k1_MAT_s001 <= 0.2
 0 <= s_k1_MAT_s002 <= 0.2
 0 <= s_k1_MAT_s003 <= 0.2
 0 <= s_k1_MAT_s004 <= 0.2
 0 <= s_k1_MAT_s005 <= 0.2
 0 <= s_k1_MAT_s006 <= 0.2
 0 <= s_k1_MAT_s007 <= 0.2
 0 <= s_k1_MAT_s010 <= 0.2
 0 <= s_k1_MAT_s011 <= 0.2
 0 <= s_k1_MAT_s012 <= 0.2
 0 <= s_k1_MAT_s013 <= 0.2
 0 <= s_k1_MAT_s014 <= 0.2
 0 <= s_k1_MAT_s015 <= 0.2
 0 <= s_k1_MAT_s016 <= 0.2
 0 <= s_k1_MAT_s017 <= 0.2
 0 <= s_k1_MAT_s100 <= 0.2
 0 <= s_k1_MAT_s101 <= 0.2
 0 <= s_k1_MAT_s102 <= 0.2
 0 <= s_k1_MAT_s103 <= 0.2
 0 <= s_k1_MAT_s104 <= 0.2
 0 <= s_k1_MAT_s105 <= 0.2
 0 <= s_k1_MAT_s106 <= 0.2
 0 <= s_k1_MAT_s107 <= 0.2
 0 <= s_k1_MAT_s110 <= 0.2
 0 <= s_k1_MAT_s111 <= 0.2
 0 <= s_k1_MAT_s112 <= 0.2
 0 <= s_k1_MAT_s113 <= 0.2
 0 <= s_k1_MAT_s114 <= 0.2
 0 <= s_k1_MAT_s115 <= 0.2
 0 <= s_k1_MAT_s116 <= 0.2
 0 <= s_k1_MAT_s117 <= 0.2
 0 <= s_k1_MSFT_s0 <= 0.2
 0 <= s_k1_MSFT_s1 <= 0.2
 0 <= s_k1_MSFT_s00 <= 0.2
 0 <= s_k1_MSFT_s01 <= 0.2
 0 <= s_k1_MSFT_s10 <= 0.2
 0 <= s_k1_MSFT_s11 <= 0.2
 0 <= s_k1_MSFT_s000 <= 0.2
 0 <= s_k1_MSFT_s001 <= 0.2
 0 <= s_k1_MSFT_s002 <= 0.2
 0 <= s_k1_MSFT_s003 <= 0.2
 0 <= s_k1_MSFT_s004 <= 0.2
 0 <= s_k1_MSFT_s005 <= 0.2
 0 <= s_k1_MSFT_s006 <= 0.2
 0 <= s_k1_MSFT_s007 <= 0.2
 0 <= s_k1_MSFT_s010 <= 0.2
 0 <= s_k1_MSFT_s011 <= 0.2
 0 <= s_k1_MSFT_s012 <= 0.2
 0 <= s_k1_MSFT_s013 <= 0.2
 0 <= s_k1_MSFT_s014 <= 0.2
 0 <= s_k1_MSFT_s015 <= 0.2
 0 <= s_k1_MSFT_s016 <= 0.2
 0 <= s_k1_MSFT_s017 <= 0.2
 0 <= s_k1_MSFT_s100 <= 0.2
 0 <= s_k1_MSFT_s101 <= 0.2
 0 <= s_k1_MSFT_s102 <= 0.2
 0 <= s_k1_MSFT_s103 <= 0.2
 0 <= s_k1_MSFT_s104 <= 0.2
 0 <= s_k1_MSFT_s105 <= 0.2
 0 <= s_k1_MSFT_s106 <= 0.2
 0 <= s_k1_MSFT_s107 <= 0.2
 0 <= s_k1_MSFT_s110 <= 0.2
 0 <= s_k1_MSFT_s111 <= 0.2
 0 <= s_k1_MSFT_s112 <= 0.2
 0 <= s_k1_MSFT_s113 <= 0.2
 0 <= s_k1_MSFT_s114 <= 0.2
 0 <= s_k1_MSFT_s115 <= 0.2
 0 <= s_k1_MSFT_s116 <= 0.2
 0 <= s_k1_MSFT_s117 <= 0.2
 0 <= s_k1_PG_s0 <= 0.2
 0 <= s_k1_PG_s1 <= 0.2
 0 <= s_k1_PG_s00 <= 0.2
 0 <= s_k1_PG_s01 <= 0.2
 0 <= s_k1_PG_s10 <= 0.2
 0 <= s_k1_PG_s11 <= 0.2
 0 <= s_k1_PG_s000 <= 0.2
 0 <= s_k1_PG_s001 <= 0.2
 0 <= s_k1_PG_s002 <= 0.2
 0 <= s_k1_PG_s003 <= 0.2
 0 <= s_k1_PG_s004 <= 0.2
 0 <= s_k1_PG_s005 <= 0.2
 0 <= s_k1_PG_s006 <= 0.2
 0 <= s_k1_PG_s007 <= 0.2
 0 <= s_k1_PG_s010 <= 0.2
 0 <= s_k1_PG_s011 <= 0.2
 0 <= s_k1_PG_s012 <= 0.2
 0 <= s_k1_PG_s013 <= 0.2
 0 <= s_k1_PG_s014 <= 0.2
 0 <= s_k1_PG_s015 <= 0.2
 0 <= s_k1_PG_s016 <= 0.2
 0 <= s_k1_PG_s017 <= 0.2
 0 <= s_k1_PG_s100 <= 0.2
 0 <= s_k1_PG_s101 <= 0.2
 0 <= s_k1_PG_s102 <= 0.2
 0 <= s_k1_PG_s103 <= 0.2
 0 <= s_k1_PG_s104 <= 0.2
 0 <= s_k1_PG_s105 <= 0.2
 0 <= s_k1_PG_s106 <= 0.2
 0 <= s_k1_PG_s107 <= 0.2
 0 <= s_k1_PG_s110 <= 0.2
 0 <= s_k1_PG_s111 <= 0.2
 0 <= s_k1_PG_s112 <= 0.2
 0 <= s_k1_PG_s113 <= 0.2
 0 <= s_k1_PG_s114 <= 0.2
 0 <= s_k1_PG_s115 <= 0.2
 0 <= s_k1_PG_s116 <= 0.2
 0 <= s_k1_PG_s117 <= 0.2
End
