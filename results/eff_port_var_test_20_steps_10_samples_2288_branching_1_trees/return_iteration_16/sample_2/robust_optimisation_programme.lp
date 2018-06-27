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
 w_k1_ADM_s0_return_constraint:  0.999156 w_0_ADM - w_k1_ADM_s0
                                 + 0.99 b_k1_ADM_s0 - 1.01 s_k1_ADM_s0  = 0
 w_k1_ADM_s1_return_constraint:  1.07188 w_0_ADM - w_k1_ADM_s1
                                 + 0.99 b_k1_ADM_s1 - 1.01 s_k1_ADM_s1  = 0
 w_k1_ADM_s00_return_constraint: 0.796798 w_k1_ADM_s0 - w_k1_ADM_s00
                                 + 0.99 b_k1_ADM_s00 - 1.01 s_k1_ADM_s00  = 0
 w_k1_ADM_s01_return_constraint: 1.16084 w_k1_ADM_s0 - w_k1_ADM_s01
                                 + 0.99 b_k1_ADM_s01 - 1.01 s_k1_ADM_s01  = 0
 w_k1_ADM_s10_return_constraint: 1.03148 w_k1_ADM_s1 - w_k1_ADM_s10
                                 + 0.99 b_k1_ADM_s10 - 1.01 s_k1_ADM_s10  = 0
 w_k1_ADM_s11_return_constraint: 0.883961 w_k1_ADM_s1 - w_k1_ADM_s11
                                 + 0.99 b_k1_ADM_s11 - 1.01 s_k1_ADM_s11  = 0
 w_k1_ADM_s000_return_constraint: 1.32552 w_k1_ADM_s00 - w_k1_ADM_s000
                                 + 0.99 b_k1_ADM_s000 - 1.01 s_k1_ADM_s000  = 0
 w_k1_ADM_s001_return_constraint: 1.24695 w_k1_ADM_s00 - w_k1_ADM_s001
                                 + 0.99 b_k1_ADM_s001 - 1.01 s_k1_ADM_s001  = 0
 w_k1_ADM_s002_return_constraint: 0.800427 w_k1_ADM_s00 - w_k1_ADM_s002
                                 + 0.99 b_k1_ADM_s002 - 1.01 s_k1_ADM_s002  = 0
 w_k1_ADM_s003_return_constraint: 1.24809 w_k1_ADM_s00 - w_k1_ADM_s003
                                 + 0.99 b_k1_ADM_s003 - 1.01 s_k1_ADM_s003  = 0
 w_k1_ADM_s004_return_constraint: 1.04478 w_k1_ADM_s00 - w_k1_ADM_s004
                                 + 0.99 b_k1_ADM_s004 - 1.01 s_k1_ADM_s004  = 0
 w_k1_ADM_s005_return_constraint: 0.918126 w_k1_ADM_s00 - w_k1_ADM_s005
                                 + 0.99 b_k1_ADM_s005 - 1.01 s_k1_ADM_s005  = 0
 w_k1_ADM_s006_return_constraint: 1.47395 w_k1_ADM_s00 - w_k1_ADM_s006
                                 + 0.99 b_k1_ADM_s006 - 1.01 s_k1_ADM_s006  = 0
 w_k1_ADM_s007_return_constraint: 0.984051 w_k1_ADM_s00 - w_k1_ADM_s007
                                 + 0.99 b_k1_ADM_s007 - 1.01 s_k1_ADM_s007  = 0
 w_k1_ADM_s010_return_constraint: 1.00548 w_k1_ADM_s01 - w_k1_ADM_s010
                                 + 0.99 b_k1_ADM_s010 - 1.01 s_k1_ADM_s010  = 0
 w_k1_ADM_s011_return_constraint: 0.88855 w_k1_ADM_s01 - w_k1_ADM_s011
                                 + 0.99 b_k1_ADM_s011 - 1.01 s_k1_ADM_s011  = 0
 w_k1_ADM_s012_return_constraint: 0.731357 w_k1_ADM_s01 - w_k1_ADM_s012
                                 + 0.99 b_k1_ADM_s012 - 1.01 s_k1_ADM_s012  = 0
 w_k1_ADM_s013_return_constraint: 1.43751 w_k1_ADM_s01 - w_k1_ADM_s013
                                 + 0.99 b_k1_ADM_s013 - 1.01 s_k1_ADM_s013  = 0
 w_k1_ADM_s014_return_constraint: 0.823639 w_k1_ADM_s01 - w_k1_ADM_s014
                                 + 0.99 b_k1_ADM_s014 - 1.01 s_k1_ADM_s014  = 0
 w_k1_ADM_s015_return_constraint: 1.18795 w_k1_ADM_s01 - w_k1_ADM_s015
                                 + 0.99 b_k1_ADM_s015 - 1.01 s_k1_ADM_s015  = 0
 w_k1_ADM_s016_return_constraint: 0.706946 w_k1_ADM_s01 - w_k1_ADM_s016
                                 + 0.99 b_k1_ADM_s016 - 1.01 s_k1_ADM_s016  = 0
 w_k1_ADM_s017_return_constraint: 1.15443 w_k1_ADM_s01 - w_k1_ADM_s017
                                 + 0.99 b_k1_ADM_s017 - 1.01 s_k1_ADM_s017  = 0
 w_k1_ADM_s100_return_constraint: 1.1745 w_k1_ADM_s10 - w_k1_ADM_s100
                                 + 0.99 b_k1_ADM_s100 - 1.01 s_k1_ADM_s100  = 0
 w_k1_ADM_s101_return_constraint: 1.44676 w_k1_ADM_s10 - w_k1_ADM_s101
                                 + 0.99 b_k1_ADM_s101 - 1.01 s_k1_ADM_s101  = 0
 w_k1_ADM_s102_return_constraint: 1.19796 w_k1_ADM_s10 - w_k1_ADM_s102
                                 + 0.99 b_k1_ADM_s102 - 1.01 s_k1_ADM_s102  = 0
 w_k1_ADM_s103_return_constraint: 0.830746 w_k1_ADM_s10 - w_k1_ADM_s103
                                 + 0.99 b_k1_ADM_s103 - 1.01 s_k1_ADM_s103  = 0
 w_k1_ADM_s104_return_constraint: 1.2253 w_k1_ADM_s10 - w_k1_ADM_s104
                                 + 0.99 b_k1_ADM_s104 - 1.01 s_k1_ADM_s104  = 0
 w_k1_ADM_s105_return_constraint: 1.65888 w_k1_ADM_s10 - w_k1_ADM_s105
                                 + 0.99 b_k1_ADM_s105 - 1.01 s_k1_ADM_s105  = 0
 w_k1_ADM_s106_return_constraint: 1.26225 w_k1_ADM_s10 - w_k1_ADM_s106
                                 + 0.99 b_k1_ADM_s106 - 1.01 s_k1_ADM_s106  = 0
 w_k1_ADM_s107_return_constraint: 0.864228 w_k1_ADM_s10 - w_k1_ADM_s107
                                 + 0.99 b_k1_ADM_s107 - 1.01 s_k1_ADM_s107  = 0
 w_k1_ADM_s110_return_constraint: 1.06618 w_k1_ADM_s11 - w_k1_ADM_s110
                                 + 0.99 b_k1_ADM_s110 - 1.01 s_k1_ADM_s110  = 0
 w_k1_ADM_s111_return_constraint: 0.836775 w_k1_ADM_s11 - w_k1_ADM_s111
                                 + 0.99 b_k1_ADM_s111 - 1.01 s_k1_ADM_s111  = 0
 w_k1_ADM_s112_return_constraint: 0.819572 w_k1_ADM_s11 - w_k1_ADM_s112
                                 + 0.99 b_k1_ADM_s112 - 1.01 s_k1_ADM_s112  = 0
 w_k1_ADM_s113_return_constraint: 1.24786 w_k1_ADM_s11 - w_k1_ADM_s113
                                 + 0.99 b_k1_ADM_s113 - 1.01 s_k1_ADM_s113  = 0
 w_k1_ADM_s114_return_constraint: 1.21604 w_k1_ADM_s11 - w_k1_ADM_s114
                                 + 0.99 b_k1_ADM_s114 - 1.01 s_k1_ADM_s114  = 0
 w_k1_ADM_s115_return_constraint: 1.09452 w_k1_ADM_s11 - w_k1_ADM_s115
                                 + 0.99 b_k1_ADM_s115 - 1.01 s_k1_ADM_s115  = 0
 w_k1_ADM_s116_return_constraint: 1.05352 w_k1_ADM_s11 - w_k1_ADM_s116
                                 + 0.99 b_k1_ADM_s116 - 1.01 s_k1_ADM_s116  = 0
 w_k1_ADM_s117_return_constraint: 0.866449 w_k1_ADM_s11 - w_k1_ADM_s117
                                 + 0.99 b_k1_ADM_s117 - 1.01 s_k1_ADM_s117  = 0
 w_k1_AXP_s0_return_constraint:  0.890569 w_0_AXP - w_k1_AXP_s0
                                 + 0.99 b_k1_AXP_s0 - 1.01 s_k1_AXP_s0  = 0
 w_k1_AXP_s1_return_constraint:  1.06119 w_0_AXP - w_k1_AXP_s1
                                 + 0.99 b_k1_AXP_s1 - 1.01 s_k1_AXP_s1  = 0
 w_k1_AXP_s00_return_constraint: 0.797188 w_k1_AXP_s0 - w_k1_AXP_s00
                                 + 0.99 b_k1_AXP_s00 - 1.01 s_k1_AXP_s00  = 0
 w_k1_AXP_s01_return_constraint: 1.26652 w_k1_AXP_s0 - w_k1_AXP_s01
                                 + 0.99 b_k1_AXP_s01 - 1.01 s_k1_AXP_s01  = 0
 w_k1_AXP_s10_return_constraint: 1.31861 w_k1_AXP_s1 - w_k1_AXP_s10
                                 + 0.99 b_k1_AXP_s10 - 1.01 s_k1_AXP_s10  = 0
 w_k1_AXP_s11_return_constraint: 0.912791 w_k1_AXP_s1 - w_k1_AXP_s11
                                 + 0.99 b_k1_AXP_s11 - 1.01 s_k1_AXP_s11  = 0
 w_k1_AXP_s000_return_constraint: 1.6541 w_k1_AXP_s00 - w_k1_AXP_s000
                                 + 0.99 b_k1_AXP_s000 - 1.01 s_k1_AXP_s000  = 0
 w_k1_AXP_s001_return_constraint: 1.08048 w_k1_AXP_s00 - w_k1_AXP_s001
                                 + 0.99 b_k1_AXP_s001 - 1.01 s_k1_AXP_s001  = 0
 w_k1_AXP_s002_return_constraint: 0.688139 w_k1_AXP_s00 - w_k1_AXP_s002
                                 + 0.99 b_k1_AXP_s002 - 1.01 s_k1_AXP_s002  = 0
 w_k1_AXP_s003_return_constraint: 1.62626 w_k1_AXP_s00 - w_k1_AXP_s003
                                 + 0.99 b_k1_AXP_s003 - 1.01 s_k1_AXP_s003  = 0
 w_k1_AXP_s004_return_constraint: 0.832286 w_k1_AXP_s00 - w_k1_AXP_s004
                                 + 0.99 b_k1_AXP_s004 - 1.01 s_k1_AXP_s004  = 0
 w_k1_AXP_s005_return_constraint: 0.839984 w_k1_AXP_s00 - w_k1_AXP_s005
                                 + 0.99 b_k1_AXP_s005 - 1.01 s_k1_AXP_s005  = 0
 w_k1_AXP_s006_return_constraint: 1.72415 w_k1_AXP_s00 - w_k1_AXP_s006
                                 + 0.99 b_k1_AXP_s006 - 1.01 s_k1_AXP_s006  = 0
 w_k1_AXP_s007_return_constraint: 1.01976 w_k1_AXP_s00 - w_k1_AXP_s007
                                 + 0.99 b_k1_AXP_s007 - 1.01 s_k1_AXP_s007  = 0
 w_k1_AXP_s010_return_constraint: 1.26357 w_k1_AXP_s01 - w_k1_AXP_s010
                                 + 0.99 b_k1_AXP_s010 - 1.01 s_k1_AXP_s010  = 0
 w_k1_AXP_s011_return_constraint: 0.607699 w_k1_AXP_s01 - w_k1_AXP_s011
                                 + 0.99 b_k1_AXP_s011 - 1.01 s_k1_AXP_s011  = 0
 w_k1_AXP_s012_return_constraint: 0.753153 w_k1_AXP_s01 - w_k1_AXP_s012
                                 + 0.99 b_k1_AXP_s012 - 1.01 s_k1_AXP_s012  = 0
 w_k1_AXP_s013_return_constraint: 1.44727 w_k1_AXP_s01 - w_k1_AXP_s013
                                 + 0.99 b_k1_AXP_s013 - 1.01 s_k1_AXP_s013  = 0
 w_k1_AXP_s014_return_constraint: 0.88971 w_k1_AXP_s01 - w_k1_AXP_s014
                                 + 0.99 b_k1_AXP_s014 - 1.01 s_k1_AXP_s014  = 0
 w_k1_AXP_s015_return_constraint: 0.956421 w_k1_AXP_s01 - w_k1_AXP_s015
                                 + 0.99 b_k1_AXP_s015 - 1.01 s_k1_AXP_s015  = 0
 w_k1_AXP_s016_return_constraint: 0.611894 w_k1_AXP_s01 - w_k1_AXP_s016
                                 + 0.99 b_k1_AXP_s016 - 1.01 s_k1_AXP_s016  = 0
 w_k1_AXP_s017_return_constraint: 0.907902 w_k1_AXP_s01 - w_k1_AXP_s017
                                 + 0.99 b_k1_AXP_s017 - 1.01 s_k1_AXP_s017  = 0
 w_k1_AXP_s100_return_constraint: 1.02421 w_k1_AXP_s10 - w_k1_AXP_s100
                                 + 0.99 b_k1_AXP_s100 - 1.01 s_k1_AXP_s100  = 0
 w_k1_AXP_s101_return_constraint: 1.53531 w_k1_AXP_s10 - w_k1_AXP_s101
                                 + 0.99 b_k1_AXP_s101 - 1.01 s_k1_AXP_s101  = 0
 w_k1_AXP_s102_return_constraint: 1.01247 w_k1_AXP_s10 - w_k1_AXP_s102
                                 + 0.99 b_k1_AXP_s102 - 1.01 s_k1_AXP_s102  = 0
 w_k1_AXP_s103_return_constraint: 0.674575 w_k1_AXP_s10 - w_k1_AXP_s103
                                 + 0.99 b_k1_AXP_s103 - 1.01 s_k1_AXP_s103  = 0
 w_k1_AXP_s104_return_constraint: 1.25842 w_k1_AXP_s10 - w_k1_AXP_s104
                                 + 0.99 b_k1_AXP_s104 - 1.01 s_k1_AXP_s104  = 0
 w_k1_AXP_s105_return_constraint: 1.63226 w_k1_AXP_s10 - w_k1_AXP_s105
                                 + 0.99 b_k1_AXP_s105 - 1.01 s_k1_AXP_s105  = 0
 w_k1_AXP_s106_return_constraint: 0.965881 w_k1_AXP_s10 - w_k1_AXP_s106
                                 + 0.99 b_k1_AXP_s106 - 1.01 s_k1_AXP_s106  = 0
 w_k1_AXP_s107_return_constraint: 0.88061 w_k1_AXP_s10 - w_k1_AXP_s107
                                 + 0.99 b_k1_AXP_s107 - 1.01 s_k1_AXP_s107  = 0
 w_k1_AXP_s110_return_constraint: 0.994606 w_k1_AXP_s11 - w_k1_AXP_s110
                                 + 0.99 b_k1_AXP_s110 - 1.01 s_k1_AXP_s110  = 0
 w_k1_AXP_s111_return_constraint: 0.838242 w_k1_AXP_s11 - w_k1_AXP_s111
                                 + 0.99 b_k1_AXP_s111 - 1.01 s_k1_AXP_s111  = 0
 w_k1_AXP_s112_return_constraint: 0.617922 w_k1_AXP_s11 - w_k1_AXP_s112
                                 + 0.99 b_k1_AXP_s112 - 1.01 s_k1_AXP_s112  = 0
 w_k1_AXP_s113_return_constraint: 1.3986 w_k1_AXP_s11 - w_k1_AXP_s113
                                 + 0.99 b_k1_AXP_s113 - 1.01 s_k1_AXP_s113  = 0
 w_k1_AXP_s114_return_constraint: 1.32921 w_k1_AXP_s11 - w_k1_AXP_s114
                                 + 0.99 b_k1_AXP_s114 - 1.01 s_k1_AXP_s114  = 0
 w_k1_AXP_s115_return_constraint: 1.08305 w_k1_AXP_s11 - w_k1_AXP_s115
                                 + 0.99 b_k1_AXP_s115 - 1.01 s_k1_AXP_s115  = 0
 w_k1_AXP_s116_return_constraint: 1.45311 w_k1_AXP_s11 - w_k1_AXP_s116
                                 + 0.99 b_k1_AXP_s116 - 1.01 s_k1_AXP_s116  = 0
 w_k1_AXP_s117_return_constraint: 0.901629 w_k1_AXP_s11 - w_k1_AXP_s117
                                 + 0.99 b_k1_AXP_s117 - 1.01 s_k1_AXP_s117  = 0
 w_k1_DIS_s0_return_constraint:  0.877136 w_0_DIS - w_k1_DIS_s0
                                 + 0.99 b_k1_DIS_s0 - 1.01 s_k1_DIS_s0  = 0
 w_k1_DIS_s1_return_constraint:  1.10259 w_0_DIS - w_k1_DIS_s1
                                 + 0.99 b_k1_DIS_s1 - 1.01 s_k1_DIS_s1  = 0
 w_k1_DIS_s00_return_constraint: 0.7152 w_k1_DIS_s0 - w_k1_DIS_s00
                                 + 0.99 b_k1_DIS_s00 - 1.01 s_k1_DIS_s00  = 0
 w_k1_DIS_s01_return_constraint: 1.25204 w_k1_DIS_s0 - w_k1_DIS_s01
                                 + 0.99 b_k1_DIS_s01 - 1.01 s_k1_DIS_s01  = 0
 w_k1_DIS_s10_return_constraint: 1.37789 w_k1_DIS_s1 - w_k1_DIS_s10
                                 + 0.99 b_k1_DIS_s10 - 1.01 s_k1_DIS_s10  = 0
 w_k1_DIS_s11_return_constraint: 0.855017 w_k1_DIS_s1 - w_k1_DIS_s11
                                 + 0.99 b_k1_DIS_s11 - 1.01 s_k1_DIS_s11  = 0
 w_k1_DIS_s000_return_constraint: 1.58748 w_k1_DIS_s00 - w_k1_DIS_s000
                                 + 0.99 b_k1_DIS_s000 - 1.01 s_k1_DIS_s000  = 0
 w_k1_DIS_s001_return_constraint: 1.33908 w_k1_DIS_s00 - w_k1_DIS_s001
                                 + 0.99 b_k1_DIS_s001 - 1.01 s_k1_DIS_s001  = 0
 w_k1_DIS_s002_return_constraint: 0.621006 w_k1_DIS_s00 - w_k1_DIS_s002
                                 + 0.99 b_k1_DIS_s002 - 1.01 s_k1_DIS_s002  = 0
 w_k1_DIS_s003_return_constraint: 1.14165 w_k1_DIS_s00 - w_k1_DIS_s003
                                 + 0.99 b_k1_DIS_s003 - 1.01 s_k1_DIS_s003  = 0
 w_k1_DIS_s004_return_constraint: 0.909401 w_k1_DIS_s00 - w_k1_DIS_s004
                                 + 0.99 b_k1_DIS_s004 - 1.01 s_k1_DIS_s004  = 0
 w_k1_DIS_s005_return_constraint: 0.850081 w_k1_DIS_s00 - w_k1_DIS_s005
                                 + 0.99 b_k1_DIS_s005 - 1.01 s_k1_DIS_s005  = 0
 w_k1_DIS_s006_return_constraint: 1.72542 w_k1_DIS_s00 - w_k1_DIS_s006
                                 + 0.99 b_k1_DIS_s006 - 1.01 s_k1_DIS_s006  = 0
 w_k1_DIS_s007_return_constraint: 1.02578 w_k1_DIS_s00 - w_k1_DIS_s007
                                 + 0.99 b_k1_DIS_s007 - 1.01 s_k1_DIS_s007  = 0
 w_k1_DIS_s010_return_constraint: 1.06481 w_k1_DIS_s01 - w_k1_DIS_s010
                                 + 0.99 b_k1_DIS_s010 - 1.01 s_k1_DIS_s010  = 0
 w_k1_DIS_s011_return_constraint: 0.728441 w_k1_DIS_s01 - w_k1_DIS_s011
                                 + 0.99 b_k1_DIS_s011 - 1.01 s_k1_DIS_s011  = 0
 w_k1_DIS_s012_return_constraint: 0.747753 w_k1_DIS_s01 - w_k1_DIS_s012
                                 + 0.99 b_k1_DIS_s012 - 1.01 s_k1_DIS_s012  = 0
 w_k1_DIS_s013_return_constraint: 1.71596 w_k1_DIS_s01 - w_k1_DIS_s013
                                 + 0.99 b_k1_DIS_s013 - 1.01 s_k1_DIS_s013  = 0
 w_k1_DIS_s014_return_constraint: 1.05732 w_k1_DIS_s01 - w_k1_DIS_s014
                                 + 0.99 b_k1_DIS_s014 - 1.01 s_k1_DIS_s014  = 0
 w_k1_DIS_s015_return_constraint: 1.08412 w_k1_DIS_s01 - w_k1_DIS_s015
                                 + 0.99 b_k1_DIS_s015 - 1.01 s_k1_DIS_s015  = 0
 w_k1_DIS_s016_return_constraint: 0.621899 w_k1_DIS_s01 - w_k1_DIS_s016
                                 + 0.99 b_k1_DIS_s016 - 1.01 s_k1_DIS_s016  = 0
 w_k1_DIS_s017_return_constraint: 1.01446 w_k1_DIS_s01 - w_k1_DIS_s017
                                 + 0.99 b_k1_DIS_s017 - 1.01 s_k1_DIS_s017  = 0
 w_k1_DIS_s100_return_constraint: 1.30095 w_k1_DIS_s10 - w_k1_DIS_s100
                                 + 0.99 b_k1_DIS_s100 - 1.01 s_k1_DIS_s100  = 0
 w_k1_DIS_s101_return_constraint: 1.43199 w_k1_DIS_s10 - w_k1_DIS_s101
                                 + 0.99 b_k1_DIS_s101 - 1.01 s_k1_DIS_s101  = 0
 w_k1_DIS_s102_return_constraint: 0.99862 w_k1_DIS_s10 - w_k1_DIS_s102
                                 + 0.99 b_k1_DIS_s102 - 1.01 s_k1_DIS_s102  = 0
 w_k1_DIS_s103_return_constraint: 0.572169 w_k1_DIS_s10 - w_k1_DIS_s103
                                 + 0.99 b_k1_DIS_s103 - 1.01 s_k1_DIS_s103  = 0
 w_k1_DIS_s104_return_constraint: 1.28975 w_k1_DIS_s10 - w_k1_DIS_s104
                                 + 0.99 b_k1_DIS_s104 - 1.01 s_k1_DIS_s104  = 0
 w_k1_DIS_s105_return_constraint: 1.95301 w_k1_DIS_s10 - w_k1_DIS_s105
                                 + 0.99 b_k1_DIS_s105 - 1.01 s_k1_DIS_s105  = 0
 w_k1_DIS_s106_return_constraint: 0.910876 w_k1_DIS_s10 - w_k1_DIS_s106
                                 + 0.99 b_k1_DIS_s106 - 1.01 s_k1_DIS_s106  = 0
 w_k1_DIS_s107_return_constraint: 0.738134 w_k1_DIS_s10 - w_k1_DIS_s107
                                 + 0.99 b_k1_DIS_s107 - 1.01 s_k1_DIS_s107  = 0
 w_k1_DIS_s110_return_constraint: 1.1631 w_k1_DIS_s11 - w_k1_DIS_s110
                                 + 0.99 b_k1_DIS_s110 - 1.01 s_k1_DIS_s110  = 0
 w_k1_DIS_s111_return_constraint: 0.822189 w_k1_DIS_s11 - w_k1_DIS_s111
                                 + 0.99 b_k1_DIS_s111 - 1.01 s_k1_DIS_s111  = 0
 w_k1_DIS_s112_return_constraint: 0.5746 w_k1_DIS_s11 - w_k1_DIS_s112
                                 + 0.99 b_k1_DIS_s112 - 1.01 s_k1_DIS_s112  = 0
 w_k1_DIS_s113_return_constraint: 1.55472 w_k1_DIS_s11 - w_k1_DIS_s113
                                 + 0.99 b_k1_DIS_s113 - 1.01 s_k1_DIS_s113  = 0
 w_k1_DIS_s114_return_constraint: 1.49899 w_k1_DIS_s11 - w_k1_DIS_s114
                                 + 0.99 b_k1_DIS_s114 - 1.01 s_k1_DIS_s114  = 0
 w_k1_DIS_s115_return_constraint: 0.995772 w_k1_DIS_s11 - w_k1_DIS_s115
                                 + 0.99 b_k1_DIS_s115 - 1.01 s_k1_DIS_s115  = 0
 w_k1_DIS_s116_return_constraint: 1.18366 w_k1_DIS_s11 - w_k1_DIS_s116
                                 + 0.99 b_k1_DIS_s116 - 1.01 s_k1_DIS_s116  = 0
 w_k1_DIS_s117_return_constraint: 0.804453 w_k1_DIS_s11 - w_k1_DIS_s117
                                 + 0.99 b_k1_DIS_s117 - 1.01 s_k1_DIS_s117  = 0
 w_k1_FDX_s0_return_constraint:  0.970441 w_0_FDX - w_k1_FDX_s0
                                 + 0.99 b_k1_FDX_s0 - 1.01 s_k1_FDX_s0  = 0
 w_k1_FDX_s1_return_constraint:  1.04821 w_0_FDX - w_k1_FDX_s1
                                 + 0.99 b_k1_FDX_s1 - 1.01 s_k1_FDX_s1  = 0
 w_k1_FDX_s00_return_constraint: 0.805132 w_k1_FDX_s0 - w_k1_FDX_s00
                                 + 0.99 b_k1_FDX_s00 - 1.01 s_k1_FDX_s00  = 0
 w_k1_FDX_s01_return_constraint: 1.19032 w_k1_FDX_s0 - w_k1_FDX_s01
                                 + 0.99 b_k1_FDX_s01 - 1.01 s_k1_FDX_s01  = 0
 w_k1_FDX_s10_return_constraint: 1.19387 w_k1_FDX_s1 - w_k1_FDX_s10
                                 + 0.99 b_k1_FDX_s10 - 1.01 s_k1_FDX_s10  = 0
 w_k1_FDX_s11_return_constraint: 0.904301 w_k1_FDX_s1 - w_k1_FDX_s11
                                 + 0.99 b_k1_FDX_s11 - 1.01 s_k1_FDX_s11  = 0
 w_k1_FDX_s000_return_constraint: 1.27139 w_k1_FDX_s00 - w_k1_FDX_s000
                                 + 0.99 b_k1_FDX_s000 - 1.01 s_k1_FDX_s000  = 0
 w_k1_FDX_s001_return_constraint: 1.15423 w_k1_FDX_s00 - w_k1_FDX_s001
                                 + 0.99 b_k1_FDX_s001 - 1.01 s_k1_FDX_s001  = 0
 w_k1_FDX_s002_return_constraint: 0.822376 w_k1_FDX_s00 - w_k1_FDX_s002
                                 + 0.99 b_k1_FDX_s002 - 1.01 s_k1_FDX_s002  = 0
 w_k1_FDX_s003_return_constraint: 1.24297 w_k1_FDX_s00 - w_k1_FDX_s003
                                 + 0.99 b_k1_FDX_s003 - 1.01 s_k1_FDX_s003  = 0
 w_k1_FDX_s004_return_constraint: 0.903731 w_k1_FDX_s00 - w_k1_FDX_s004
                                 + 0.99 b_k1_FDX_s004 - 1.01 s_k1_FDX_s004  = 0
 w_k1_FDX_s005_return_constraint: 0.887113 w_k1_FDX_s00 - w_k1_FDX_s005
                                 + 0.99 b_k1_FDX_s005 - 1.01 s_k1_FDX_s005  = 0
 w_k1_FDX_s006_return_constraint: 1.50767 w_k1_FDX_s00 - w_k1_FDX_s006
                                 + 0.99 b_k1_FDX_s006 - 1.01 s_k1_FDX_s006  = 0
 w_k1_FDX_s007_return_constraint: 0.889708 w_k1_FDX_s00 - w_k1_FDX_s007
                                 + 0.99 b_k1_FDX_s007 - 1.01 s_k1_FDX_s007  = 0
 w_k1_FDX_s010_return_constraint: 1.02483 w_k1_FDX_s01 - w_k1_FDX_s010
                                 + 0.99 b_k1_FDX_s010 - 1.01 s_k1_FDX_s010  = 0
 w_k1_FDX_s011_return_constraint: 0.857032 w_k1_FDX_s01 - w_k1_FDX_s011
                                 + 0.99 b_k1_FDX_s011 - 1.01 s_k1_FDX_s011  = 0
 w_k1_FDX_s012_return_constraint: 0.716638 w_k1_FDX_s01 - w_k1_FDX_s012
                                 + 0.99 b_k1_FDX_s012 - 1.01 s_k1_FDX_s012  = 0
 w_k1_FDX_s013_return_constraint: 1.38963 w_k1_FDX_s01 - w_k1_FDX_s013
                                 + 0.99 b_k1_FDX_s013 - 1.01 s_k1_FDX_s013  = 0
 w_k1_FDX_s014_return_constraint: 1.01481 w_k1_FDX_s01 - w_k1_FDX_s014
                                 + 0.99 b_k1_FDX_s014 - 1.01 s_k1_FDX_s014  = 0
 w_k1_FDX_s015_return_constraint: 0.9307 w_k1_FDX_s01 - w_k1_FDX_s015
                                 + 0.99 b_k1_FDX_s015 - 1.01 s_k1_FDX_s015  = 0
 w_k1_FDX_s016_return_constraint: 0.74994 w_k1_FDX_s01 - w_k1_FDX_s016
                                 + 0.99 b_k1_FDX_s016 - 1.01 s_k1_FDX_s016  = 0
 w_k1_FDX_s017_return_constraint: 1.12249 w_k1_FDX_s01 - w_k1_FDX_s017
                                 + 0.99 b_k1_FDX_s017 - 1.01 s_k1_FDX_s017  = 0
 w_k1_FDX_s100_return_constraint: 1.17799 w_k1_FDX_s10 - w_k1_FDX_s100
                                 + 0.99 b_k1_FDX_s100 - 1.01 s_k1_FDX_s100  = 0
 w_k1_FDX_s101_return_constraint: 1.31272 w_k1_FDX_s10 - w_k1_FDX_s101
                                 + 0.99 b_k1_FDX_s101 - 1.01 s_k1_FDX_s101  = 0
 w_k1_FDX_s102_return_constraint: 0.961057 w_k1_FDX_s10 - w_k1_FDX_s102
                                 + 0.99 b_k1_FDX_s102 - 1.01 s_k1_FDX_s102  = 0
 w_k1_FDX_s103_return_constraint: 0.717921 w_k1_FDX_s10 - w_k1_FDX_s103
                                 + 0.99 b_k1_FDX_s103 - 1.01 s_k1_FDX_s103  = 0
 w_k1_FDX_s104_return_constraint: 1.09047 w_k1_FDX_s10 - w_k1_FDX_s104
                                 + 0.99 b_k1_FDX_s104 - 1.01 s_k1_FDX_s104  = 0
 w_k1_FDX_s105_return_constraint: 1.54194 w_k1_FDX_s10 - w_k1_FDX_s105
                                 + 0.99 b_k1_FDX_s105 - 1.01 s_k1_FDX_s105  = 0
 w_k1_FDX_s106_return_constraint: 1.06558 w_k1_FDX_s10 - w_k1_FDX_s106
                                 + 0.99 b_k1_FDX_s106 - 1.01 s_k1_FDX_s106  = 0
 w_k1_FDX_s107_return_constraint: 0.754019 w_k1_FDX_s10 - w_k1_FDX_s107
                                 + 0.99 b_k1_FDX_s107 - 1.01 s_k1_FDX_s107  = 0
 w_k1_FDX_s110_return_constraint: 1.06045 w_k1_FDX_s11 - w_k1_FDX_s110
                                 + 0.99 b_k1_FDX_s110 - 1.01 s_k1_FDX_s110  = 0
 w_k1_FDX_s111_return_constraint: 0.936976 w_k1_FDX_s11 - w_k1_FDX_s111
                                 + 0.99 b_k1_FDX_s111 - 1.01 s_k1_FDX_s111  = 0
 w_k1_FDX_s112_return_constraint: 0.700819 w_k1_FDX_s11 - w_k1_FDX_s112
                                 + 0.99 b_k1_FDX_s112 - 1.01 s_k1_FDX_s112  = 0
 w_k1_FDX_s113_return_constraint: 1.27933 w_k1_FDX_s11 - w_k1_FDX_s113
                                 + 0.99 b_k1_FDX_s113 - 1.01 s_k1_FDX_s113  = 0
 w_k1_FDX_s114_return_constraint: 1.37232 w_k1_FDX_s11 - w_k1_FDX_s114
                                 + 0.99 b_k1_FDX_s114 - 1.01 s_k1_FDX_s114  = 0
 w_k1_FDX_s115_return_constraint: 1.10901 w_k1_FDX_s11 - w_k1_FDX_s115
                                 + 0.99 b_k1_FDX_s115 - 1.01 s_k1_FDX_s115  = 0
 w_k1_FDX_s116_return_constraint: 1.06717 w_k1_FDX_s11 - w_k1_FDX_s116
                                 + 0.99 b_k1_FDX_s116 - 1.01 s_k1_FDX_s116  = 0
 w_k1_FDX_s117_return_constraint: 0.790885 w_k1_FDX_s11 - w_k1_FDX_s117
                                 + 0.99 b_k1_FDX_s117 - 1.01 s_k1_FDX_s117  = 0
 w_k1_IBM_s0_return_constraint:  0.768236 w_0_IBM - w_k1_IBM_s0
                                 + 0.99 b_k1_IBM_s0 - 1.01 s_k1_IBM_s0  = 0
 w_k1_IBM_s1_return_constraint:  1.1306 w_0_IBM - w_k1_IBM_s1
                                 + 0.99 b_k1_IBM_s1 - 1.01 s_k1_IBM_s1  = 0
 w_k1_IBM_s00_return_constraint: 0.945081 w_k1_IBM_s0 - w_k1_IBM_s00
                                 + 0.99 b_k1_IBM_s00 - 1.01 s_k1_IBM_s00  = 0
 w_k1_IBM_s01_return_constraint: 1.062 w_k1_IBM_s0 - w_k1_IBM_s01
                                 + 0.99 b_k1_IBM_s01 - 1.01 s_k1_IBM_s01  = 0
 w_k1_IBM_s10_return_constraint: 1.3014 w_k1_IBM_s1 - w_k1_IBM_s10
                                 + 0.99 b_k1_IBM_s10 - 1.01 s_k1_IBM_s10  = 0
 w_k1_IBM_s11_return_constraint: 0.844867 w_k1_IBM_s1 - w_k1_IBM_s11
                                 + 0.99 b_k1_IBM_s11 - 1.01 s_k1_IBM_s11  = 0
 w_k1_IBM_s000_return_constraint: 1.60691 w_k1_IBM_s00 - w_k1_IBM_s000
                                 + 0.99 b_k1_IBM_s000 - 1.01 s_k1_IBM_s000  = 0
 w_k1_IBM_s001_return_constraint: 0.881158 w_k1_IBM_s00 - w_k1_IBM_s001
                                 + 0.99 b_k1_IBM_s001 - 1.01 s_k1_IBM_s001  = 0
 w_k1_IBM_s002_return_constraint: 0.592971 w_k1_IBM_s00 - w_k1_IBM_s002
                                 + 0.99 b_k1_IBM_s002 - 1.01 s_k1_IBM_s002  = 0
 w_k1_IBM_s003_return_constraint: 1.22553 w_k1_IBM_s00 - w_k1_IBM_s003
                                 + 0.99 b_k1_IBM_s003 - 1.01 s_k1_IBM_s003  = 0
 w_k1_IBM_s004_return_constraint: 0.834309 w_k1_IBM_s00 - w_k1_IBM_s004
                                 + 0.99 b_k1_IBM_s004 - 1.01 s_k1_IBM_s004  = 0
 w_k1_IBM_s005_return_constraint: 1.06306 w_k1_IBM_s00 - w_k1_IBM_s005
                                 + 0.99 b_k1_IBM_s005 - 1.01 s_k1_IBM_s005  = 0
 w_k1_IBM_s006_return_constraint: 1.04783 w_k1_IBM_s00 - w_k1_IBM_s006
                                 + 0.99 b_k1_IBM_s006 - 1.01 s_k1_IBM_s006  = 0
 w_k1_IBM_s007_return_constraint: 1.44438 w_k1_IBM_s00 - w_k1_IBM_s007
                                 + 0.99 b_k1_IBM_s007 - 1.01 s_k1_IBM_s007  = 0
 w_k1_IBM_s010_return_constraint: 1.23018 w_k1_IBM_s01 - w_k1_IBM_s010
                                 + 0.99 b_k1_IBM_s010 - 1.01 s_k1_IBM_s010  = 0
 w_k1_IBM_s011_return_constraint: 0.575542 w_k1_IBM_s01 - w_k1_IBM_s011
                                 + 0.99 b_k1_IBM_s011 - 1.01 s_k1_IBM_s011  = 0
 w_k1_IBM_s012_return_constraint: 1.06566 w_k1_IBM_s01 - w_k1_IBM_s012
                                 + 0.99 b_k1_IBM_s012 - 1.01 s_k1_IBM_s012  = 0
 w_k1_IBM_s013_return_constraint: 1.14124 w_k1_IBM_s01 - w_k1_IBM_s013
                                 + 0.99 b_k1_IBM_s013 - 1.01 s_k1_IBM_s013  = 0
 w_k1_IBM_s014_return_constraint: 0.918099 w_k1_IBM_s01 - w_k1_IBM_s014
                                 + 0.99 b_k1_IBM_s014 - 1.01 s_k1_IBM_s014  = 0
 w_k1_IBM_s015_return_constraint: 1.56071 w_k1_IBM_s01 - w_k1_IBM_s015
                                 + 0.99 b_k1_IBM_s015 - 1.01 s_k1_IBM_s015  = 0
 w_k1_IBM_s016_return_constraint: 0.754337 w_k1_IBM_s01 - w_k1_IBM_s016
                                 + 0.99 b_k1_IBM_s016 - 1.01 s_k1_IBM_s016  = 0
 w_k1_IBM_s017_return_constraint: 0.632695 w_k1_IBM_s01 - w_k1_IBM_s017
                                 + 0.99 b_k1_IBM_s017 - 1.01 s_k1_IBM_s017  = 0
 w_k1_IBM_s100_return_constraint: 0.710958 w_k1_IBM_s10 - w_k1_IBM_s100
                                 + 0.99 b_k1_IBM_s100 - 1.01 s_k1_IBM_s100  = 0
 w_k1_IBM_s101_return_constraint: 1.15843 w_k1_IBM_s10 - w_k1_IBM_s101
                                 + 0.99 b_k1_IBM_s101 - 1.01 s_k1_IBM_s101  = 0
 w_k1_IBM_s102_return_constraint: 1.15184 w_k1_IBM_s10 - w_k1_IBM_s102
                                 + 0.99 b_k1_IBM_s102 - 1.01 s_k1_IBM_s102  = 0
 w_k1_IBM_s103_return_constraint: 0.894897 w_k1_IBM_s10 - w_k1_IBM_s103
                                 + 0.99 b_k1_IBM_s103 - 1.01 s_k1_IBM_s103  = 0
 w_k1_IBM_s104_return_constraint: 1.74973 w_k1_IBM_s10 - w_k1_IBM_s104
                                 + 0.99 b_k1_IBM_s104 - 1.01 s_k1_IBM_s104  = 0
 w_k1_IBM_s105_return_constraint: 1.15018 w_k1_IBM_s10 - w_k1_IBM_s105
                                 + 0.99 b_k1_IBM_s105 - 1.01 s_k1_IBM_s105  = 0
 w_k1_IBM_s106_return_constraint: 0.77398 w_k1_IBM_s10 - w_k1_IBM_s106
                                 + 0.99 b_k1_IBM_s106 - 1.01 s_k1_IBM_s106  = 0
 w_k1_IBM_s107_return_constraint: 1.24279 w_k1_IBM_s10 - w_k1_IBM_s107
                                 + 0.99 b_k1_IBM_s107 - 1.01 s_k1_IBM_s107  = 0
 w_k1_IBM_s110_return_constraint: 1.16703 w_k1_IBM_s11 - w_k1_IBM_s110
                                 + 0.99 b_k1_IBM_s110 - 1.01 s_k1_IBM_s110  = 0
 w_k1_IBM_s111_return_constraint: 0.806345 w_k1_IBM_s11 - w_k1_IBM_s111
                                 + 0.99 b_k1_IBM_s111 - 1.01 s_k1_IBM_s111  = 0
 w_k1_IBM_s112_return_constraint: 0.844469 w_k1_IBM_s11 - w_k1_IBM_s112
                                 + 0.99 b_k1_IBM_s112 - 1.01 s_k1_IBM_s112  = 0
 w_k1_IBM_s113_return_constraint: 1.48078 w_k1_IBM_s11 - w_k1_IBM_s113
                                 + 0.99 b_k1_IBM_s113 - 1.01 s_k1_IBM_s113  = 0
 w_k1_IBM_s114_return_constraint: 0.962404 w_k1_IBM_s11 - w_k1_IBM_s114
                                 + 0.99 b_k1_IBM_s114 - 1.01 s_k1_IBM_s114  = 0
 w_k1_IBM_s115_return_constraint: 0.799544 w_k1_IBM_s11 - w_k1_IBM_s115
                                 + 0.99 b_k1_IBM_s115 - 1.01 s_k1_IBM_s115  = 0
 w_k1_IBM_s116_return_constraint: 1.78499 w_k1_IBM_s11 - w_k1_IBM_s116
                                 + 0.99 b_k1_IBM_s116 - 1.01 s_k1_IBM_s116  = 0
 w_k1_IBM_s117_return_constraint: 1.35697 w_k1_IBM_s11 - w_k1_IBM_s117
                                 + 0.99 b_k1_IBM_s117 - 1.01 s_k1_IBM_s117  = 0
 w_k1_INTC_s0_return_constraint: 0.903315 w_0_INTC - w_k1_INTC_s0
                                 + 0.99 b_k1_INTC_s0 - 1.01 s_k1_INTC_s0  = 0
 w_k1_INTC_s1_return_constraint: 1.00861 w_0_INTC - w_k1_INTC_s1
                                 + 0.99 b_k1_INTC_s1 - 1.01 s_k1_INTC_s1  = 0
 w_k1_INTC_s00_return_constraint: 0.859151 w_k1_INTC_s0 - w_k1_INTC_s00
                                 + 0.99 b_k1_INTC_s00 - 1.01 s_k1_INTC_s00  = 0
 w_k1_INTC_s01_return_constraint: 0.995798 w_k1_INTC_s0 - w_k1_INTC_s01
                                 + 0.99 b_k1_INTC_s01 - 1.01 s_k1_INTC_s01  = 0
 w_k1_INTC_s10_return_constraint: 1.15824 w_k1_INTC_s1 - w_k1_INTC_s10
                                 + 0.99 b_k1_INTC_s10 - 1.01 s_k1_INTC_s10  = 0
 w_k1_INTC_s11_return_constraint: 0.910155 w_k1_INTC_s1 - w_k1_INTC_s11
                                 + 0.99 b_k1_INTC_s11 - 1.01 s_k1_INTC_s11  = 0
 w_k1_INTC_s000_return_constraint: 1.45046 w_k1_INTC_s00 - w_k1_INTC_s000
                                 + 0.99 b_k1_INTC_s000 - 1.01 s_k1_INTC_s000
                                  = 0
 w_k1_INTC_s001_return_constraint: 1.14813 w_k1_INTC_s00 - w_k1_INTC_s001
                                 + 0.99 b_k1_INTC_s001 - 1.01 s_k1_INTC_s001
                                  = 0
 w_k1_INTC_s002_return_constraint: 0.959702 w_k1_INTC_s00 - w_k1_INTC_s002
                                 + 0.99 b_k1_INTC_s002 - 1.01 s_k1_INTC_s002
                                  = 0
 w_k1_INTC_s003_return_constraint: 1.27902 w_k1_INTC_s00 - w_k1_INTC_s003
                                 + 0.99 b_k1_INTC_s003 - 1.01 s_k1_INTC_s003
                                  = 0
 w_k1_INTC_s004_return_constraint: 0.907643 w_k1_INTC_s00 - w_k1_INTC_s004
                                 + 0.99 b_k1_INTC_s004 - 1.01 s_k1_INTC_s004
                                  = 0
 w_k1_INTC_s005_return_constraint: 1.02614 w_k1_INTC_s00 - w_k1_INTC_s005
                                 + 0.99 b_k1_INTC_s005 - 1.01 s_k1_INTC_s005
                                  = 0
 w_k1_INTC_s006_return_constraint: 1.59363 w_k1_INTC_s00 - w_k1_INTC_s006
                                 + 0.99 b_k1_INTC_s006 - 1.01 s_k1_INTC_s006
                                  = 0
 w_k1_INTC_s007_return_constraint: 0.903261 w_k1_INTC_s00 - w_k1_INTC_s007
                                 + 0.99 b_k1_INTC_s007 - 1.01 s_k1_INTC_s007
                                  = 0
 w_k1_INTC_s010_return_constraint: 1.10337 w_k1_INTC_s01 - w_k1_INTC_s010
                                 + 0.99 b_k1_INTC_s010 - 1.01 s_k1_INTC_s010
                                  = 0
 w_k1_INTC_s011_return_constraint: 1.05512 w_k1_INTC_s01 - w_k1_INTC_s011
                                 + 0.99 b_k1_INTC_s011 - 1.01 s_k1_INTC_s011
                                  = 0
 w_k1_INTC_s012_return_constraint: 1.10021 w_k1_INTC_s01 - w_k1_INTC_s012
                                 + 0.99 b_k1_INTC_s012 - 1.01 s_k1_INTC_s012
                                  = 0
 w_k1_INTC_s013_return_constraint: 1.51201 w_k1_INTC_s01 - w_k1_INTC_s013
                                 + 0.99 b_k1_INTC_s013 - 1.01 s_k1_INTC_s013
                                  = 0
 w_k1_INTC_s014_return_constraint: 1.20391 w_k1_INTC_s01 - w_k1_INTC_s014
                                 + 0.99 b_k1_INTC_s014 - 1.01 s_k1_INTC_s014
                                  = 0
 w_k1_INTC_s015_return_constraint: 1.22773 w_k1_INTC_s01 - w_k1_INTC_s015
                                 + 0.99 b_k1_INTC_s015 - 1.01 s_k1_INTC_s015
                                  = 0
 w_k1_INTC_s016_return_constraint: 0.949828 w_k1_INTC_s01 - w_k1_INTC_s016
                                 + 0.99 b_k1_INTC_s016 - 1.01 s_k1_INTC_s016
                                  = 0
 w_k1_INTC_s017_return_constraint: 1.3069 w_k1_INTC_s01 - w_k1_INTC_s017
                                 + 0.99 b_k1_INTC_s017 - 1.01 s_k1_INTC_s017
                                  = 0
 w_k1_INTC_s100_return_constraint: 1.15965 w_k1_INTC_s10 - w_k1_INTC_s100
                                 + 0.99 b_k1_INTC_s100 - 1.01 s_k1_INTC_s100
                                  = 0
 w_k1_INTC_s101_return_constraint: 1.27165 w_k1_INTC_s10 - w_k1_INTC_s101
                                 + 0.99 b_k1_INTC_s101 - 1.01 s_k1_INTC_s101
                                  = 0
 w_k1_INTC_s102_return_constraint: 0.935298 w_k1_INTC_s10 - w_k1_INTC_s102
                                 + 0.99 b_k1_INTC_s102 - 1.01 s_k1_INTC_s102
                                  = 0
 w_k1_INTC_s103_return_constraint: 0.765957 w_k1_INTC_s10 - w_k1_INTC_s103
                                 + 0.99 b_k1_INTC_s103 - 1.01 s_k1_INTC_s103
                                  = 0
 w_k1_INTC_s104_return_constraint: 1.00705 w_k1_INTC_s10 - w_k1_INTC_s104
                                 + 0.99 b_k1_INTC_s104 - 1.01 s_k1_INTC_s104
                                  = 0
 w_k1_INTC_s105_return_constraint: 1.38767 w_k1_INTC_s10 - w_k1_INTC_s105
                                 + 0.99 b_k1_INTC_s105 - 1.01 s_k1_INTC_s105
                                  = 0
 w_k1_INTC_s106_return_constraint: 0.931773 w_k1_INTC_s10 - w_k1_INTC_s106
                                 + 0.99 b_k1_INTC_s106 - 1.01 s_k1_INTC_s106
                                  = 0
 w_k1_INTC_s107_return_constraint: 0.866528 w_k1_INTC_s10 - w_k1_INTC_s107
                                 + 0.99 b_k1_INTC_s107 - 1.01 s_k1_INTC_s107
                                  = 0
 w_k1_INTC_s110_return_constraint: 1.13278 w_k1_INTC_s11 - w_k1_INTC_s110
                                 + 0.99 b_k1_INTC_s110 - 1.01 s_k1_INTC_s110
                                  = 0
 w_k1_INTC_s111_return_constraint: 0.929285 w_k1_INTC_s11 - w_k1_INTC_s111
                                 + 0.99 b_k1_INTC_s111 - 1.01 s_k1_INTC_s111
                                  = 0
 w_k1_INTC_s112_return_constraint: 0.755544 w_k1_INTC_s11 - w_k1_INTC_s112
                                 + 0.99 b_k1_INTC_s112 - 1.01 s_k1_INTC_s112
                                  = 0
 w_k1_INTC_s113_return_constraint: 1.33488 w_k1_INTC_s11 - w_k1_INTC_s113
                                 + 0.99 b_k1_INTC_s113 - 1.01 s_k1_INTC_s113
                                  = 0
 w_k1_INTC_s114_return_constraint: 1.38164 w_k1_INTC_s11 - w_k1_INTC_s114
                                 + 0.99 b_k1_INTC_s114 - 1.01 s_k1_INTC_s114
                                  = 0
 w_k1_INTC_s115_return_constraint: 0.99843 w_k1_INTC_s11 - w_k1_INTC_s115
                                 + 0.99 b_k1_INTC_s115 - 1.01 s_k1_INTC_s115
                                  = 0
 w_k1_INTC_s116_return_constraint: 1.0805 w_k1_INTC_s11 - w_k1_INTC_s116
                                 + 0.99 b_k1_INTC_s116 - 1.01 s_k1_INTC_s116
                                  = 0
 w_k1_INTC_s117_return_constraint: 0.895775 w_k1_INTC_s11 - w_k1_INTC_s117
                                 + 0.99 b_k1_INTC_s117 - 1.01 s_k1_INTC_s117
                                  = 0
 w_k1_KO_s0_return_constraint:   0.816193 w_0_KO - w_k1_KO_s0 + 0.99 b_k1_KO_s0
                                 - 1.01 s_k1_KO_s0  = 0
 w_k1_KO_s1_return_constraint:   1.11824 w_0_KO - w_k1_KO_s1 + 0.99 b_k1_KO_s1
                                 - 1.01 s_k1_KO_s1  = 0
 w_k1_KO_s00_return_constraint:  0.876089 w_k1_KO_s0 - w_k1_KO_s00
                                 + 0.99 b_k1_KO_s00 - 1.01 s_k1_KO_s00  = 0
 w_k1_KO_s01_return_constraint:  1.17175 w_k1_KO_s0 - w_k1_KO_s01
                                 + 0.99 b_k1_KO_s01 - 1.01 s_k1_KO_s01  = 0
 w_k1_KO_s10_return_constraint:  1.17662 w_k1_KO_s1 - w_k1_KO_s10
                                 + 0.99 b_k1_KO_s10 - 1.01 s_k1_KO_s10  = 0
 w_k1_KO_s11_return_constraint:  0.869283 w_k1_KO_s1 - w_k1_KO_s11
                                 + 0.99 b_k1_KO_s11 - 1.01 s_k1_KO_s11  = 0
 w_k1_KO_s000_return_constraint: 1.44674 w_k1_KO_s00 - w_k1_KO_s000
                                 + 0.99 b_k1_KO_s000 - 1.01 s_k1_KO_s000  = 0
 w_k1_KO_s001_return_constraint: 1.18435 w_k1_KO_s00 - w_k1_KO_s001
                                 + 0.99 b_k1_KO_s001 - 1.01 s_k1_KO_s001  = 0
 w_k1_KO_s002_return_constraint: 0.730609 w_k1_KO_s00 - w_k1_KO_s002
                                 + 0.99 b_k1_KO_s002 - 1.01 s_k1_KO_s002  = 0
 w_k1_KO_s003_return_constraint: 1.03973 w_k1_KO_s00 - w_k1_KO_s003
                                 + 0.99 b_k1_KO_s003 - 1.01 s_k1_KO_s003  = 0
 w_k1_KO_s004_return_constraint: 0.955466 w_k1_KO_s00 - w_k1_KO_s004
                                 + 0.99 b_k1_KO_s004 - 1.01 s_k1_KO_s004  = 0
 w_k1_KO_s005_return_constraint: 1.03193 w_k1_KO_s00 - w_k1_KO_s005
                                 + 0.99 b_k1_KO_s005 - 1.01 s_k1_KO_s005  = 0
 w_k1_KO_s006_return_constraint: 1.25892 w_k1_KO_s00 - w_k1_KO_s006
                                 + 0.99 b_k1_KO_s006 - 1.01 s_k1_KO_s006  = 0
 w_k1_KO_s007_return_constraint: 1.0403 w_k1_KO_s00 - w_k1_KO_s007
                                 + 0.99 b_k1_KO_s007 - 1.01 s_k1_KO_s007  = 0
 w_k1_KO_s010_return_constraint: 1.06413 w_k1_KO_s01 - w_k1_KO_s010
                                 + 0.99 b_k1_KO_s010 - 1.01 s_k1_KO_s010  = 0
 w_k1_KO_s011_return_constraint: 0.786093 w_k1_KO_s01 - w_k1_KO_s011
                                 + 0.99 b_k1_KO_s011 - 1.01 s_k1_KO_s011  = 0
 w_k1_KO_s012_return_constraint: 0.919933 w_k1_KO_s01 - w_k1_KO_s012
                                 + 0.99 b_k1_KO_s012 - 1.01 s_k1_KO_s012  = 0
 w_k1_KO_s013_return_constraint: 1.26033 w_k1_KO_s01 - w_k1_KO_s013
                                 + 0.99 b_k1_KO_s013 - 1.01 s_k1_KO_s013  = 0
 w_k1_KO_s014_return_constraint: 1.0719 w_k1_KO_s01 - w_k1_KO_s014
                                 + 0.99 b_k1_KO_s014 - 1.01 s_k1_KO_s014  = 0
 w_k1_KO_s015_return_constraint: 1.27809 w_k1_KO_s01 - w_k1_KO_s015
                                 + 0.99 b_k1_KO_s015 - 1.01 s_k1_KO_s015  = 0
 w_k1_KO_s016_return_constraint: 0.751753 w_k1_KO_s01 - w_k1_KO_s016
                                 + 0.99 b_k1_KO_s016 - 1.01 s_k1_KO_s016  = 0
 w_k1_KO_s017_return_constraint: 0.906868 w_k1_KO_s01 - w_k1_KO_s017
                                 + 0.99 b_k1_KO_s017 - 1.01 s_k1_KO_s017  = 0
 w_k1_KO_s100_return_constraint: 0.943208 w_k1_KO_s10 - w_k1_KO_s100
                                 + 0.99 b_k1_KO_s100 - 1.01 s_k1_KO_s100  = 0
 w_k1_KO_s101_return_constraint: 1.31054 w_k1_KO_s10 - w_k1_KO_s101
                                 + 0.99 b_k1_KO_s101 - 1.01 s_k1_KO_s101  = 0
 w_k1_KO_s102_return_constraint: 1.14565 w_k1_KO_s10 - w_k1_KO_s102
                                 + 0.99 b_k1_KO_s102 - 1.01 s_k1_KO_s102  = 0
 w_k1_KO_s103_return_constraint: 0.802775 w_k1_KO_s10 - w_k1_KO_s103
                                 + 0.99 b_k1_KO_s103 - 1.01 s_k1_KO_s103  = 0
 w_k1_KO_s104_return_constraint: 1.40462 w_k1_KO_s10 - w_k1_KO_s104
                                 + 0.99 b_k1_KO_s104 - 1.01 s_k1_KO_s104  = 0
 w_k1_KO_s105_return_constraint: 1.48625 w_k1_KO_s10 - w_k1_KO_s105
                                 + 0.99 b_k1_KO_s105 - 1.01 s_k1_KO_s105  = 0
 w_k1_KO_s106_return_constraint: 0.797534 w_k1_KO_s10 - w_k1_KO_s106
                                 + 0.99 b_k1_KO_s106 - 1.01 s_k1_KO_s106  = 0
 w_k1_KO_s107_return_constraint: 0.951407 w_k1_KO_s10 - w_k1_KO_s107
                                 + 0.99 b_k1_KO_s107 - 1.01 s_k1_KO_s107  = 0
 w_k1_KO_s110_return_constraint: 1.10478 w_k1_KO_s11 - w_k1_KO_s110
                                 + 0.99 b_k1_KO_s110 - 1.01 s_k1_KO_s110  = 0
 w_k1_KO_s111_return_constraint: 0.847163 w_k1_KO_s11 - w_k1_KO_s111
                                 + 0.99 b_k1_KO_s111 - 1.01 s_k1_KO_s111  = 0
 w_k1_KO_s112_return_constraint: 0.751491 w_k1_KO_s11 - w_k1_KO_s112
                                 + 0.99 b_k1_KO_s112 - 1.01 s_k1_KO_s112  = 0
 w_k1_KO_s113_return_constraint: 1.25209 w_k1_KO_s11 - w_k1_KO_s113
                                 + 0.99 b_k1_KO_s113 - 1.01 s_k1_KO_s113  = 0
 w_k1_KO_s114_return_constraint: 1.10975 w_k1_KO_s11 - w_k1_KO_s114
                                 + 0.99 b_k1_KO_s114 - 1.01 s_k1_KO_s114  = 0
 w_k1_KO_s115_return_constraint: 0.889799 w_k1_KO_s11 - w_k1_KO_s115
                                 + 0.99 b_k1_KO_s115 - 1.01 s_k1_KO_s115  = 0
 w_k1_KO_s116_return_constraint: 1.30567 w_k1_KO_s11 - w_k1_KO_s116
                                 + 0.99 b_k1_KO_s116 - 1.01 s_k1_KO_s116  = 0
 w_k1_KO_s117_return_constraint: 0.979711 w_k1_KO_s11 - w_k1_KO_s117
                                 + 0.99 b_k1_KO_s117 - 1.01 s_k1_KO_s117  = 0
 w_k1_MAT_s0_return_constraint:  0.752588 w_0_MAT - w_k1_MAT_s0
                                 + 0.99 b_k1_MAT_s0 - 1.01 s_k1_MAT_s0  = 0
 w_k1_MAT_s1_return_constraint:  1.13564 w_0_MAT - w_k1_MAT_s1
                                 + 0.99 b_k1_MAT_s1 - 1.01 s_k1_MAT_s1  = 0
 w_k1_MAT_s00_return_constraint: 1.01468 w_k1_MAT_s0 - w_k1_MAT_s00
                                 + 0.99 b_k1_MAT_s00 - 1.01 s_k1_MAT_s00  = 0
 w_k1_MAT_s01_return_constraint: 1.13667 w_k1_MAT_s0 - w_k1_MAT_s01
                                 + 0.99 b_k1_MAT_s01 - 1.01 s_k1_MAT_s01  = 0
 w_k1_MAT_s10_return_constraint: 1.33824 w_k1_MAT_s1 - w_k1_MAT_s10
                                 + 0.99 b_k1_MAT_s10 - 1.01 s_k1_MAT_s10  = 0
 w_k1_MAT_s11_return_constraint: 0.835304 w_k1_MAT_s1 - w_k1_MAT_s11
                                 + 0.99 b_k1_MAT_s11 - 1.01 s_k1_MAT_s11  = 0
 w_k1_MAT_s000_return_constraint: 1.4284 w_k1_MAT_s00 - w_k1_MAT_s000
                                 + 0.99 b_k1_MAT_s000 - 1.01 s_k1_MAT_s000  = 0
 w_k1_MAT_s001_return_constraint: 0.784183 w_k1_MAT_s00 - w_k1_MAT_s001
                                 + 0.99 b_k1_MAT_s001 - 1.01 s_k1_MAT_s001  = 0
 w_k1_MAT_s002_return_constraint: 0.562611 w_k1_MAT_s00 - w_k1_MAT_s002
                                 + 0.99 b_k1_MAT_s002 - 1.01 s_k1_MAT_s002  = 0
 w_k1_MAT_s003_return_constraint: 1.20217 w_k1_MAT_s00 - w_k1_MAT_s003
                                 + 0.99 b_k1_MAT_s003 - 1.01 s_k1_MAT_s003  = 0
 w_k1_MAT_s004_return_constraint: 0.576574 w_k1_MAT_s00 - w_k1_MAT_s004
                                 + 0.99 b_k1_MAT_s004 - 1.01 s_k1_MAT_s004  = 0
 w_k1_MAT_s005_return_constraint: 0.922581 w_k1_MAT_s00 - w_k1_MAT_s005
                                 + 0.99 b_k1_MAT_s005 - 1.01 s_k1_MAT_s005  = 0
 w_k1_MAT_s006_return_constraint: 1.06852 w_k1_MAT_s00 - w_k1_MAT_s006
                                 + 0.99 b_k1_MAT_s006 - 1.01 s_k1_MAT_s006  = 0
 w_k1_MAT_s007_return_constraint: 1.08165 w_k1_MAT_s00 - w_k1_MAT_s007
                                 + 0.99 b_k1_MAT_s007 - 1.01 s_k1_MAT_s007  = 0
 w_k1_MAT_s010_return_constraint: 1.38331 w_k1_MAT_s01 - w_k1_MAT_s010
                                 + 0.99 b_k1_MAT_s010 - 1.01 s_k1_MAT_s010  = 0
 w_k1_MAT_s011_return_constraint: 0.523449 w_k1_MAT_s01 - w_k1_MAT_s011
                                 + 0.99 b_k1_MAT_s011 - 1.01 s_k1_MAT_s011  = 0
 w_k1_MAT_s012_return_constraint: 0.950483 w_k1_MAT_s01 - w_k1_MAT_s012
                                 + 0.99 b_k1_MAT_s012 - 1.01 s_k1_MAT_s012  = 0
 w_k1_MAT_s013_return_constraint: 1.2264 w_k1_MAT_s01 - w_k1_MAT_s013
                                 + 0.99 b_k1_MAT_s013 - 1.01 s_k1_MAT_s013  = 0
 w_k1_MAT_s014_return_constraint: 0.919143 w_k1_MAT_s01 - w_k1_MAT_s014
                                 + 0.99 b_k1_MAT_s014 - 1.01 s_k1_MAT_s014  = 0
 w_k1_MAT_s015_return_constraint: 1.47734 w_k1_MAT_s01 - w_k1_MAT_s015
                                 + 0.99 b_k1_MAT_s015 - 1.01 s_k1_MAT_s015  = 0
 w_k1_MAT_s016_return_constraint: 0.62561 w_k1_MAT_s01 - w_k1_MAT_s016
                                 + 0.99 b_k1_MAT_s016 - 1.01 s_k1_MAT_s016  = 0
 w_k1_MAT_s017_return_constraint: 0.594085 w_k1_MAT_s01 - w_k1_MAT_s017
                                 + 0.99 b_k1_MAT_s017 - 1.01 s_k1_MAT_s017  = 0
 w_k1_MAT_s100_return_constraint: 0.698134 w_k1_MAT_s10 - w_k1_MAT_s100
                                 + 0.99 b_k1_MAT_s100 - 1.01 s_k1_MAT_s100  = 0
 w_k1_MAT_s101_return_constraint: 1.84675 w_k1_MAT_s10 - w_k1_MAT_s101
                                 + 0.99 b_k1_MAT_s101 - 1.01 s_k1_MAT_s101  = 0
 w_k1_MAT_s102_return_constraint: 1.26644 w_k1_MAT_s10 - w_k1_MAT_s102
                                 + 0.99 b_k1_MAT_s102 - 1.01 s_k1_MAT_s102  = 0
 w_k1_MAT_s103_return_constraint: 0.69636 w_k1_MAT_s10 - w_k1_MAT_s103
                                 + 0.99 b_k1_MAT_s103 - 1.01 s_k1_MAT_s103  = 0
 w_k1_MAT_s104_return_constraint: 1.81798 w_k1_MAT_s10 - w_k1_MAT_s104
                                 + 0.99 b_k1_MAT_s104 - 1.01 s_k1_MAT_s104  = 0
 w_k1_MAT_s105_return_constraint: 1.31615 w_k1_MAT_s10 - w_k1_MAT_s105
                                 + 0.99 b_k1_MAT_s105 - 1.01 s_k1_MAT_s105  = 0
 w_k1_MAT_s106_return_constraint: 0.811926 w_k1_MAT_s10 - w_k1_MAT_s106
                                 + 0.99 b_k1_MAT_s106 - 1.01 s_k1_MAT_s106  = 0
 w_k1_MAT_s107_return_constraint: 1.24017 w_k1_MAT_s10 - w_k1_MAT_s107
                                 + 0.99 b_k1_MAT_s107 - 1.01 s_k1_MAT_s107  = 0
 w_k1_MAT_s110_return_constraint: 1.14625 w_k1_MAT_s11 - w_k1_MAT_s110
                                 + 0.99 b_k1_MAT_s110 - 1.01 s_k1_MAT_s110  = 0
 w_k1_MAT_s111_return_constraint: 0.679796 w_k1_MAT_s11 - w_k1_MAT_s111
                                 + 0.99 b_k1_MAT_s111 - 1.01 s_k1_MAT_s111  = 0
 w_k1_MAT_s112_return_constraint: 0.811943 w_k1_MAT_s11 - w_k1_MAT_s112
                                 + 0.99 b_k1_MAT_s112 - 1.01 s_k1_MAT_s112  = 0
 w_k1_MAT_s113_return_constraint: 1.45522 w_k1_MAT_s11 - w_k1_MAT_s113
                                 + 0.99 b_k1_MAT_s113 - 1.01 s_k1_MAT_s113  = 0
 w_k1_MAT_s114_return_constraint: 1.05369 w_k1_MAT_s11 - w_k1_MAT_s114
                                 + 0.99 b_k1_MAT_s114 - 1.01 s_k1_MAT_s114  = 0
 w_k1_MAT_s115_return_constraint: 0.931933 w_k1_MAT_s11 - w_k1_MAT_s115
                                 + 0.99 b_k1_MAT_s115 - 1.01 s_k1_MAT_s115  = 0
 w_k1_MAT_s116_return_constraint: 2.05025 w_k1_MAT_s11 - w_k1_MAT_s116
                                 + 0.99 b_k1_MAT_s116 - 1.01 s_k1_MAT_s116  = 0
 w_k1_MAT_s117_return_constraint: 1.42561 w_k1_MAT_s11 - w_k1_MAT_s117
                                 + 0.99 b_k1_MAT_s117 - 1.01 s_k1_MAT_s117  = 0
 w_k1_MSFT_s0_return_constraint: 0.961453 w_0_MSFT - w_k1_MSFT_s0
                                 + 0.99 b_k1_MSFT_s0 - 1.01 s_k1_MSFT_s0  = 0
 w_k1_MSFT_s1_return_constraint: 1.07108 w_0_MSFT - w_k1_MSFT_s1
                                 + 0.99 b_k1_MSFT_s1 - 1.01 s_k1_MSFT_s1  = 0
 w_k1_MSFT_s00_return_constraint: 0.79581 w_k1_MSFT_s0 - w_k1_MSFT_s00
                                 + 0.99 b_k1_MSFT_s00 - 1.01 s_k1_MSFT_s00  = 0
 w_k1_MSFT_s01_return_constraint: 1.17587 w_k1_MSFT_s0 - w_k1_MSFT_s01
                                 + 0.99 b_k1_MSFT_s01 - 1.01 s_k1_MSFT_s01  = 0
 w_k1_MSFT_s10_return_constraint: 1.1738 w_k1_MSFT_s1 - w_k1_MSFT_s10
                                 + 0.99 b_k1_MSFT_s10 - 1.01 s_k1_MSFT_s10  = 0
 w_k1_MSFT_s11_return_constraint: 0.892608 w_k1_MSFT_s1 - w_k1_MSFT_s11
                                 + 0.99 b_k1_MSFT_s11 - 1.01 s_k1_MSFT_s11  = 0
 w_k1_MSFT_s000_return_constraint: 1.27559 w_k1_MSFT_s00 - w_k1_MSFT_s000
                                 + 0.99 b_k1_MSFT_s000 - 1.01 s_k1_MSFT_s000
                                  = 0
 w_k1_MSFT_s001_return_constraint: 1.20737 w_k1_MSFT_s00 - w_k1_MSFT_s001
                                 + 0.99 b_k1_MSFT_s001 - 1.01 s_k1_MSFT_s001
                                  = 0
 w_k1_MSFT_s002_return_constraint: 0.884951 w_k1_MSFT_s00 - w_k1_MSFT_s002
                                 + 0.99 b_k1_MSFT_s002 - 1.01 s_k1_MSFT_s002
                                  = 0
 w_k1_MSFT_s003_return_constraint: 0.993209 w_k1_MSFT_s00 - w_k1_MSFT_s003
                                 + 0.99 b_k1_MSFT_s003 - 1.01 s_k1_MSFT_s003
                                  = 0
 w_k1_MSFT_s004_return_constraint: 0.967536 w_k1_MSFT_s00 - w_k1_MSFT_s004
                                 + 0.99 b_k1_MSFT_s004 - 1.01 s_k1_MSFT_s004
                                  = 0
 w_k1_MSFT_s005_return_constraint: 0.999064 w_k1_MSFT_s00 - w_k1_MSFT_s005
                                 + 0.99 b_k1_MSFT_s005 - 1.01 s_k1_MSFT_s005
                                  = 0
 w_k1_MSFT_s006_return_constraint: 1.45071 w_k1_MSFT_s00 - w_k1_MSFT_s006
                                 + 0.99 b_k1_MSFT_s006 - 1.01 s_k1_MSFT_s006
                                  = 0
 w_k1_MSFT_s007_return_constraint: 0.915743 w_k1_MSFT_s00 - w_k1_MSFT_s007
                                 + 0.99 b_k1_MSFT_s007 - 1.01 s_k1_MSFT_s007
                                  = 0
 w_k1_MSFT_s010_return_constraint: 1.01312 w_k1_MSFT_s01 - w_k1_MSFT_s010
                                 + 0.99 b_k1_MSFT_s010 - 1.01 s_k1_MSFT_s010
                                  = 0
 w_k1_MSFT_s011_return_constraint: 0.90747 w_k1_MSFT_s01 - w_k1_MSFT_s011
                                 + 0.99 b_k1_MSFT_s011 - 1.01 s_k1_MSFT_s011
                                  = 0
 w_k1_MSFT_s012_return_constraint: 0.75647 w_k1_MSFT_s01 - w_k1_MSFT_s012
                                 + 0.99 b_k1_MSFT_s012 - 1.01 s_k1_MSFT_s012
                                  = 0
 w_k1_MSFT_s013_return_constraint: 1.27096 w_k1_MSFT_s01 - w_k1_MSFT_s013
                                 + 0.99 b_k1_MSFT_s013 - 1.01 s_k1_MSFT_s013
                                  = 0
 w_k1_MSFT_s014_return_constraint: 1.00494 w_k1_MSFT_s01 - w_k1_MSFT_s014
                                 + 0.99 b_k1_MSFT_s014 - 1.01 s_k1_MSFT_s014
                                  = 0
 w_k1_MSFT_s015_return_constraint: 1.02203 w_k1_MSFT_s01 - w_k1_MSFT_s015
                                 + 0.99 b_k1_MSFT_s015 - 1.01 s_k1_MSFT_s015
                                  = 0
 w_k1_MSFT_s016_return_constraint: 0.788679 w_k1_MSFT_s01 - w_k1_MSFT_s016
                                 + 0.99 b_k1_MSFT_s016 - 1.01 s_k1_MSFT_s016
                                  = 0
 w_k1_MSFT_s017_return_constraint: 1.08685 w_k1_MSFT_s01 - w_k1_MSFT_s017
                                 + 0.99 b_k1_MSFT_s017 - 1.01 s_k1_MSFT_s017
                                  = 0
 w_k1_MSFT_s100_return_constraint: 1.13633 w_k1_MSFT_s10 - w_k1_MSFT_s100
                                 + 0.99 b_k1_MSFT_s100 - 1.01 s_k1_MSFT_s100
                                  = 0
 w_k1_MSFT_s101_return_constraint: 1.23588 w_k1_MSFT_s10 - w_k1_MSFT_s101
                                 + 0.99 b_k1_MSFT_s101 - 1.01 s_k1_MSFT_s101
                                  = 0
 w_k1_MSFT_s102_return_constraint: 0.919064 w_k1_MSFT_s10 - w_k1_MSFT_s102
                                 + 0.99 b_k1_MSFT_s102 - 1.01 s_k1_MSFT_s102
                                  = 0
 w_k1_MSFT_s103_return_constraint: 0.706167 w_k1_MSFT_s10 - w_k1_MSFT_s103
                                 + 0.99 b_k1_MSFT_s103 - 1.01 s_k1_MSFT_s103
                                  = 0
 w_k1_MSFT_s104_return_constraint: 1.02112 w_k1_MSFT_s10 - w_k1_MSFT_s104
                                 + 0.99 b_k1_MSFT_s104 - 1.01 s_k1_MSFT_s104
                                  = 0
 w_k1_MSFT_s105_return_constraint: 1.42426 w_k1_MSFT_s10 - w_k1_MSFT_s105
                                 + 0.99 b_k1_MSFT_s105 - 1.01 s_k1_MSFT_s105
                                  = 0
 w_k1_MSFT_s106_return_constraint: 0.963964 w_k1_MSFT_s10 - w_k1_MSFT_s106
                                 + 0.99 b_k1_MSFT_s106 - 1.01 s_k1_MSFT_s106
                                  = 0
 w_k1_MSFT_s107_return_constraint: 0.753344 w_k1_MSFT_s10 - w_k1_MSFT_s107
                                 + 0.99 b_k1_MSFT_s107 - 1.01 s_k1_MSFT_s107
                                  = 0
 w_k1_MSFT_s110_return_constraint: 1.10222 w_k1_MSFT_s11 - w_k1_MSFT_s110
                                 + 0.99 b_k1_MSFT_s110 - 1.01 s_k1_MSFT_s110
                                  = 0
 w_k1_MSFT_s111_return_constraint: 0.957611 w_k1_MSFT_s11 - w_k1_MSFT_s111
                                 + 0.99 b_k1_MSFT_s111 - 1.01 s_k1_MSFT_s111
                                  = 0
 w_k1_MSFT_s112_return_constraint: 0.750974 w_k1_MSFT_s11 - w_k1_MSFT_s112
                                 + 0.99 b_k1_MSFT_s112 - 1.01 s_k1_MSFT_s112
                                  = 0
 w_k1_MSFT_s113_return_constraint: 1.13933 w_k1_MSFT_s11 - w_k1_MSFT_s113
                                 + 0.99 b_k1_MSFT_s113 - 1.01 s_k1_MSFT_s113
                                  = 0
 w_k1_MSFT_s114_return_constraint: 1.26601 w_k1_MSFT_s11 - w_k1_MSFT_s114
                                 + 0.99 b_k1_MSFT_s114 - 1.01 s_k1_MSFT_s114
                                  = 0
 w_k1_MSFT_s115_return_constraint: 1.01298 w_k1_MSFT_s11 - w_k1_MSFT_s115
                                 + 0.99 b_k1_MSFT_s115 - 1.01 s_k1_MSFT_s115
                                  = 0
 w_k1_MSFT_s116_return_constraint: 1.10135 w_k1_MSFT_s11 - w_k1_MSFT_s116
                                 + 0.99 b_k1_MSFT_s116 - 1.01 s_k1_MSFT_s116
                                  = 0
 w_k1_MSFT_s117_return_constraint: 0.815316 w_k1_MSFT_s11 - w_k1_MSFT_s117
                                 + 0.99 b_k1_MSFT_s117 - 1.01 s_k1_MSFT_s117
                                  = 0
 w_k1_PG_s0_return_constraint:   0.933455 w_0_PG - w_k1_PG_s0 + 0.99 b_k1_PG_s0
                                 - 1.01 s_k1_PG_s0  = 0
 w_k1_PG_s1_return_constraint:   1.04621 w_0_PG - w_k1_PG_s1 + 0.99 b_k1_PG_s1
                                 - 1.01 s_k1_PG_s1  = 0
 w_k1_PG_s00_return_constraint:  0.879096 w_k1_PG_s0 - w_k1_PG_s00
                                 + 0.99 b_k1_PG_s00 - 1.01 s_k1_PG_s00  = 0
 w_k1_PG_s01_return_constraint:  1.09227 w_k1_PG_s0 - w_k1_PG_s01
                                 + 0.99 b_k1_PG_s01 - 1.01 s_k1_PG_s01  = 0
 w_k1_PG_s10_return_constraint:  1.14765 w_k1_PG_s1 - w_k1_PG_s10
                                 + 0.99 b_k1_PG_s10 - 1.01 s_k1_PG_s10  = 0
 w_k1_PG_s11_return_constraint:  0.917649 w_k1_PG_s1 - w_k1_PG_s11
                                 + 0.99 b_k1_PG_s11 - 1.01 s_k1_PG_s11  = 0
 w_k1_PG_s000_return_constraint: 1.24004 w_k1_PG_s00 - w_k1_PG_s000
                                 + 0.99 b_k1_PG_s000 - 1.01 s_k1_PG_s000  = 0
 w_k1_PG_s001_return_constraint: 1.07639 w_k1_PG_s00 - w_k1_PG_s001
                                 + 0.99 b_k1_PG_s001 - 1.01 s_k1_PG_s001  = 0
 w_k1_PG_s002_return_constraint: 0.824711 w_k1_PG_s00 - w_k1_PG_s002
                                 + 0.99 b_k1_PG_s002 - 1.01 s_k1_PG_s002  = 0
 w_k1_PG_s003_return_constraint: 1.11993 w_k1_PG_s00 - w_k1_PG_s003
                                 + 0.99 b_k1_PG_s003 - 1.01 s_k1_PG_s003  = 0
 w_k1_PG_s004_return_constraint: 0.947533 w_k1_PG_s00 - w_k1_PG_s004
                                 + 0.99 b_k1_PG_s004 - 1.01 s_k1_PG_s004  = 0
 w_k1_PG_s005_return_constraint: 0.949247 w_k1_PG_s00 - w_k1_PG_s005
                                 + 0.99 b_k1_PG_s005 - 1.01 s_k1_PG_s005  = 0
 w_k1_PG_s006_return_constraint: 1.23094 w_k1_PG_s00 - w_k1_PG_s006
                                 + 0.99 b_k1_PG_s006 - 1.01 s_k1_PG_s006  = 0
 w_k1_PG_s007_return_constraint: 1.02071 w_k1_PG_s00 - w_k1_PG_s007
                                 + 0.99 b_k1_PG_s007 - 1.01 s_k1_PG_s007  = 0
 w_k1_PG_s010_return_constraint: 1.04977 w_k1_PG_s01 - w_k1_PG_s010
                                 + 0.99 b_k1_PG_s010 - 1.01 s_k1_PG_s010  = 0
 w_k1_PG_s011_return_constraint: 0.821809 w_k1_PG_s01 - w_k1_PG_s011
                                 + 0.99 b_k1_PG_s011 - 1.01 s_k1_PG_s011  = 0
 w_k1_PG_s012_return_constraint: 0.89597 w_k1_PG_s01 - w_k1_PG_s012
                                 + 0.99 b_k1_PG_s012 - 1.01 s_k1_PG_s012  = 0
 w_k1_PG_s013_return_constraint: 1.28624 w_k1_PG_s01 - w_k1_PG_s013
                                 + 0.99 b_k1_PG_s013 - 1.01 s_k1_PG_s013  = 0
 w_k1_PG_s014_return_constraint: 1.00197 w_k1_PG_s01 - w_k1_PG_s014
                                 + 0.99 b_k1_PG_s014 - 1.01 s_k1_PG_s014  = 0
 w_k1_PG_s015_return_constraint: 1.09491 w_k1_PG_s01 - w_k1_PG_s015
                                 + 0.99 b_k1_PG_s015 - 1.01 s_k1_PG_s015  = 0
 w_k1_PG_s016_return_constraint: 0.797613 w_k1_PG_s01 - w_k1_PG_s016
                                 + 0.99 b_k1_PG_s016 - 1.01 s_k1_PG_s016  = 0
 w_k1_PG_s017_return_constraint: 1.01643 w_k1_PG_s01 - w_k1_PG_s017
                                 + 0.99 b_k1_PG_s017 - 1.01 s_k1_PG_s017  = 0
 w_k1_PG_s100_return_constraint: 1.06655 w_k1_PG_s10 - w_k1_PG_s100
                                 + 0.99 b_k1_PG_s100 - 1.01 s_k1_PG_s100  = 0
 w_k1_PG_s101_return_constraint: 1.30708 w_k1_PG_s10 - w_k1_PG_s101
                                 + 0.99 b_k1_PG_s101 - 1.01 s_k1_PG_s101  = 0
 w_k1_PG_s102_return_constraint: 1.0088 w_k1_PG_s10 - w_k1_PG_s102
                                 + 0.99 b_k1_PG_s102 - 1.01 s_k1_PG_s102  = 0
 w_k1_PG_s103_return_constraint: 0.813712 w_k1_PG_s10 - w_k1_PG_s103
                                 + 0.99 b_k1_PG_s103 - 1.01 s_k1_PG_s103  = 0
 w_k1_PG_s104_return_constraint: 1.24367 w_k1_PG_s10 - w_k1_PG_s104
                                 + 0.99 b_k1_PG_s104 - 1.01 s_k1_PG_s104  = 0
 w_k1_PG_s105_return_constraint: 1.25527 w_k1_PG_s10 - w_k1_PG_s105
                                 + 0.99 b_k1_PG_s105 - 1.01 s_k1_PG_s105  = 0
 w_k1_PG_s106_return_constraint: 0.923173 w_k1_PG_s10 - w_k1_PG_s106
                                 + 0.99 b_k1_PG_s106 - 1.01 s_k1_PG_s106  = 0
 w_k1_PG_s107_return_constraint: 0.963912 w_k1_PG_s10 - w_k1_PG_s107
                                 + 0.99 b_k1_PG_s107 - 1.01 s_k1_PG_s107  = 0
 w_k1_PG_s110_return_constraint: 1.02533 w_k1_PG_s11 - w_k1_PG_s110
                                 + 0.99 b_k1_PG_s110 - 1.01 s_k1_PG_s110  = 0
 w_k1_PG_s111_return_constraint: 0.968731 w_k1_PG_s11 - w_k1_PG_s111
                                 + 0.99 b_k1_PG_s111 - 1.01 s_k1_PG_s111  = 0
 w_k1_PG_s112_return_constraint: 0.852054 w_k1_PG_s11 - w_k1_PG_s112
                                 + 0.99 b_k1_PG_s112 - 1.01 s_k1_PG_s112  = 0
 w_k1_PG_s113_return_constraint: 1.20674 w_k1_PG_s11 - w_k1_PG_s113
                                 + 0.99 b_k1_PG_s113 - 1.01 s_k1_PG_s113  = 0
 w_k1_PG_s114_return_constraint: 1.1178 w_k1_PG_s11 - w_k1_PG_s114
                                 + 0.99 b_k1_PG_s114 - 1.01 s_k1_PG_s114  = 0
 w_k1_PG_s115_return_constraint: 0.973787 w_k1_PG_s11 - w_k1_PG_s115
                                 + 0.99 b_k1_PG_s115 - 1.01 s_k1_PG_s115  = 0
 w_k1_PG_s116_return_constraint: 1.13863 w_k1_PG_s11 - w_k1_PG_s116
                                 + 0.99 b_k1_PG_s116 - 1.01 s_k1_PG_s116  = 0
 w_k1_PG_s117_return_constraint: 0.970804 w_k1_PG_s11 - w_k1_PG_s117
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
 k1_wcvar_constraint:            wcvar - var_1 - 0.101000139882961 z_k1_s0000
                                 - 0.108000211639285 z_k1_s0001
                                 - 0.182000450262617 z_k1_s0002
                                 - 0.0910001240311804 z_k1_s0003
                                 - 0.0640001216714232 z_k1_s0004
                                 - 0.145999705715336 z_k1_s0005
                                 - 0.152999878571787 z_k1_s0006
                                 - 0.166000344019665 z_k1_s0007
                                 - 0.234000033945355 z_k1_s0010
                                 - 0.367001238769996 z_k1_s0011
                                 - 0.425000109845201 z_k1_s0012
                                 - 0.433999156942983 z_k1_s0013
                                 - 0.37300060350185 z_k1_s0014
                                 - 0.274001061425027 z_k1_s0015
                                 - 0.25300077886323 z_k1_s0016
                                 - 0.146000079010356 z_k1_s0017
                                 - 0.371998869394407 z_k1_s0020
                                 - 0.0829999777344381 z_k1_s0021
                                 - 0.309999057828672 z_k1_s0022
                                 - 0.364999093604431 z_k1_s0023
                                 - 0.249999781344914 z_k1_s0024
                                 - 0.268998573632865 z_k1_s0025
                                 - 0.294999238707733 z_k1_s0026
                                 - 0.152999832631136 z_k1_s0027
                                 - 0.0900003702797226 z_k1_s0030
                                 - 0.0480000488824572 z_k1_s0031
                                 - 0.160999993689005 z_k1_s0032
                                 - 0.0890002918654727 z_k1_s0033
                                 - 0.0780001723090314 z_k1_s0034
                                 - 0.0960000977649144 z_k1_s0035
                                 - 0.0890002918654727 z_k1_s0036
                                 - 0.0919997841079147 z_k1_s0037
                                 - 0.638999544220925 z_k1_s0040
                                 - 0.152999675039164 z_k1_s0041
                                 - 0.148999862147654 z_k1_s0042
                                 - 0.177999718759132 z_k1_s0043
                                 - 0.342999087872412 z_k1_s0044
                                 - 0.268999612284245 z_k1_s0045
                                 - 0.638999544220925 z_k1_s0046
                                 - 0.183999821195774 z_k1_s0047
                                 - 0.107000109409641 z_k1_s0050
                                 - 0.211000340962122 z_k1_s0051
                                 - 0.127999836910319 z_k1_s0052
                                 - 0.0970000893853664 z_k1_s0053
                                 - 0.232000650963355 z_k1_s0054
                                 - 0.148000575959534 z_k1_s0055
                                 - 0.0880001063135523 z_k1_s0056
                                 - 0.154000331673855 z_k1_s0057
                                 - 0.147999748327214 z_k1_s0060
                                 - 0.0419999693360871 z_k1_s0061
                                 - 0.0539999011464184 z_k1_s0062
                                 - 0.0900000085773036 z_k1_s0063
                                 - 0.0640000176549674 z_k1_s0064
                                 - 0.0410000096852141 z_k1_s0065
                                 - 0.0419999693360871 z_k1_s0066
                                 - 0.0389999863835043 z_k1_s0067
                                 - 0.182999498124658 z_k1_s0070
                                 - 0.193999050329776 z_k1_s0071
                                 - 0.109999703842132 z_k1_s0072
                                 - 0.332998897637082 z_k1_s0073
                                 - 0.292999346166479 z_k1_s0074
                                 - 0.0809998487763347 z_k1_s0075
                                 - 0.176999349104536 z_k1_s0076
                                 - 0.0709998888088398 z_k1_s0077
                                 - 0.231000063157363 z_k1_s0100
                                 - 0.108999939630176 z_k1_s0101
                                 - 0.139999880477327 z_k1_s0102
                                 - 0.247999842376932 z_k1_s0103
                                 - 0.124999898919037 z_k1_s0104
                                 - 0.198999828979916 z_k1_s0105
                                 - 0.362999648390534 z_k1_s0106
                                 - 0.162999368280459 z_k1_s0107
                                 - 0.09099996436042 z_k1_s0110
                                 - 0.1889994895951 z_k1_s0111
                                 - 0.127000058854557 z_k1_s0112
                                 - 0.206000174716771 z_k1_s0113
                                 - 0.0459999377426686 z_k1_s0114
                                 - 0.0739999197381886 z_k1_s0115
                                 - 0.102999629858786 z_k1_s0116
                                 - 0.0789999328623743 z_k1_s0117
                                 - 0.189999891698805 z_k1_s0120
                                 - 0.21600038887835 z_k1_s0121
                                 - 0.182999383689141 z_k1_s0122
                                 - 0.212999302588989 z_k1_s0123
                                 - 0.160999524229206 z_k1_s0124
                                 - 0.12499986474925 z_k1_s0125
                                 - 0.0759999566795221 z_k1_s0126
                                 - 0.0520000215596438 z_k1_s0127
                                 - 0.153999701487478 z_k1_s0130
                                 - 0.30899911043046 z_k1_s0131
                                 - 0.41299981856104 z_k1_s0132
                                 - 0.289000637326189 z_k1_s0133
                                 - 0.353998917910709 z_k1_s0134
                                 - 0.246000677156367 z_k1_s0135
                                 - 0.265999150260638 z_k1_s0136
                                 - 0.130999727819892 z_k1_s0137
                                 - 0.368998710284287 z_k1_s0140
                                 - 0.864996579412562 z_k1_s0141
                                 - 0.440997545756428 z_k1_s0142
                                 - 0.249999456703794 z_k1_s0143
                                 - 0.179999583874787 z_k1_s0144
                                 - 0.368998710284287 z_k1_s0145
                                 - 0.321998604075239 z_k1_s0146
                                 - 0.322999800842981 z_k1_s0147
                                 - 0.448000963356821 z_k1_s0150
                                 - 0.184000305189272 z_k1_s0151
                                 - 0.337000801777458 z_k1_s0152
                                 - 0.125000119715406 z_k1_s0153
                                 - 0.318001023093077 z_k1_s0154
                                 - 0.125000119715406 z_k1_s0155
                                 - 0.54200035326975 z_k1_s0156
                                 - 0.224000251378063 z_k1_s0157
                                 - 0.134999789109581 z_k1_s0160
                                 - 0.0859999260882171 z_k1_s0161
                                 - 0.363999981442623 z_k1_s0162
                                 - 0.267999583810127 z_k1_s0163
                                 - 0.10999985899664 z_k1_s0164
                                 - 0.151999741586381 z_k1_s0165
                                 - 0.274000233036037 z_k1_s0166
                                 - 0.27599989444567 z_k1_s0167
                                 - 0.193000107198826 z_k1_s0170
                                 - 0.073999974959137 z_k1_s0171
                                 - 0.149999736268526 z_k1_s0172
                                 - 0.0899998417611157 z_k1_s0173
                                 - 0.0869999258357433 z_k1_s0174
                                 - 0.0500000171561728 z_k1_s0175
                                 - 0.0590000013322844 z_k1_s0176
                                 - 0.0849999818854949 z_k1_s0177
                                 - 0.334000624157581 z_k1_s1000
                                 - 0.430000239303582 z_k1_s1001
                                 - 0.382002176733947 z_k1_s1002
                                 - 0.33800052087183 z_k1_s1003
                                 - 0.385000092515763 z_k1_s1004
                                 - 0.43200001316037 z_k1_s1005
                                 - 0.873000753679649 z_k1_s1006
                                 - 0.316000565442453 z_k1_s1007
                                 - 0.312999716466419 z_k1_s1010
                                 - 0.194000133847995 z_k1_s1011
                                 - 0.204999891695208 z_k1_s1012
                                 - 0.0829999640278109 z_k1_s1013
                                 - 0.239000464585737 z_k1_s1014
                                 - 0.142999920511782 z_k1_s1015
                                 - 0.168000265754867 z_k1_s1016
                                 - 0.270000369427502 z_k1_s1017
                                 - 1.44699989344343 z_k1_s1020
                                 - 0.763000928927532 z_k1_s1021
                                 - 0.906002289316976 z_k1_s1022
                                 - 0.495000206268621 z_k1_s1023
                                 - 0.312000307393655 z_k1_s1024
                                 - 0.587999870549182 z_k1_s1025
                                 - 0.337000071447487 z_k1_s1026
                                 - 0.57200002155473 z_k1_s1027
                                 - 0.559999694116059 z_k1_s1030
                                 - 1.26499652410667 z_k1_s1031
                                 - 1.04900102580789 z_k1_s1032
                                 - 0.560999831124756 z_k1_s1033
                                 - 1.34399724539974 z_k1_s1034
                                 - 0.30000006163347 z_k1_s1035
                                 - 0.804996640068561 z_k1_s1036
                                 - 0.429999562174951 z_k1_s1037
                                 - 0.17299990028866 z_k1_s1040
                                 - 0.230999829434601 z_k1_s1041
                                 - 0.290001053228507 z_k1_s1042
                                 - 0.24999965576702 z_k1_s1043
                                 - 0.587000190502113 z_k1_s1044
                                 - 0.374999905900171 z_k1_s1045
                                 - 0.163999789947152 z_k1_s1046
                                 - 0.745000376054527 z_k1_s1047
                                 - 0.322999999858549 z_k1_s1050
                                 - 0.221999856998974 z_k1_s1051
                                 - 0.162999693025683 z_k1_s1052
                                 - 0.184999661032816 z_k1_s1053
                                 - 0.177999701185046 z_k1_s1054
                                 - 0.257998645703336 z_k1_s1055
                                 - 0.705998713556575 z_k1_s1056
                                 - 0.162999693025683 z_k1_s1057
                                 - 0.5430003459714 z_k1_s1060
                                 - 0.746001379092358 z_k1_s1061
                                 - 0.415000347156861 z_k1_s1062
                                 - 0.484000607518082 z_k1_s1063
                                 - 0.680999714692701 z_k1_s1064
                                 - 0.489000292471463 z_k1_s1065
                                 - 1.8420037449442 z_k1_s1066
                                 - 0.560000426814036 z_k1_s1067
                                 - 0.528000027795052 z_k1_s1070
                                 - 0.153999775023932 z_k1_s1071
                                 - 0.158999684172896 z_k1_s1072
                                 - 0.593997933522811 z_k1_s1073
                                 - 0.488000195204244 z_k1_s1074
                                 - 0.133999858728528 z_k1_s1075
                                 - 0.319998660726457 z_k1_s1076
                                 - 0.419999640701223 z_k1_s1077
                                 - 0.703001981836843 z_k1_s1100
                                 - 0.801005654229458 z_k1_s1101
                                 - 0.336001194212669 z_k1_s1102
                                 - 1.43600867931912 z_k1_s1103
                                 - 0.501001906356917 z_k1_s1104
                                 - 0.342001550739299 z_k1_s1105
                                 - 0.442002017893537 z_k1_s1106
                                 - 1.30500275272881 z_k1_s1107
                                 - 1.89700518500465 z_k1_s1110
                                 - 0.439001836506781 z_k1_s1111
                                 - 1.15100002495052 z_k1_s1112
                                 - 1.14300510299341 z_k1_s1113
                                 - 0.510001278420591 z_k1_s1114
                                 - 1.07600564146634 z_k1_s1115
                                 - 1.06800232148197 z_k1_s1116
                                 - 1.11400671484438 z_k1_s1117
                                 - 1.56300158703832 z_k1_s1120
                                 - 2.27100362556039 z_k1_s1121
                                 - 0.83800135002719 z_k1_s1122
                                 - 0.966999924943548 z_k1_s1123
                                 - 0.990003497866504 z_k1_s1124
                                 - 1.14900411751787 z_k1_s1125
                                 - 0.472001040714271 z_k1_s1126
                                 - 1.29900001767304 z_k1_s1127
                                 - 0.380001487955657 z_k1_s1130
                                 - 0.19800032609206 z_k1_s1131
                                 - 0.282000490297819 z_k1_s1132
                                 - 0.303001455975543 z_k1_s1133
                                 - 0.595999878053355 z_k1_s1134
                                 - 0.137000311841596 z_k1_s1135
                                 - 0.573999462487043 z_k1_s1136
                                 - 0.374999971006791 z_k1_s1137
                                 - 0.970997573460009 z_k1_s1140
                                 - 0.931001083200474 z_k1_s1141
                                 - 0.603998713231389 z_k1_s1142
                                 - 0.91299583969524 z_k1_s1143
                                 - 1.32099738667924 z_k1_s1144
                                 - 0.441998922803995 z_k1_s1145
                                 - 0.834999451794621 z_k1_s1146
                                 - 1.16499742889017 z_k1_s1147
                                 - 0.671001477589723 z_k1_s1150
                                 - 0.47700152171442 z_k1_s1151
                                 - 0.205000083789914 z_k1_s1152
                                 - 0.577999930638804 z_k1_s1153
                                 - 0.209000020777018 z_k1_s1154
                                 - 0.139999902949171 z_k1_s1155
                                 - 0.316001045982728 z_k1_s1156
                                 - 0.415999470660336 z_k1_s1157
                                 - 0.157000037842268 z_k1_s1160
                                 - 0.19400026687282 z_k1_s1161
                                 - 0.139000223600659 z_k1_s1162
                                 - 0.366999815462878 z_k1_s1163
                                 - 0.165000129816482 z_k1_s1164
                                 - 0.447000735205013 z_k1_s1165
                                 - 0.599000463113159 z_k1_s1166
                                 - 0.176000228230999 z_k1_s1167
                                 - 0.686002900193613 z_k1_s1170
                                 - 1.05900169584061 z_k1_s1171
                                 - 0.595000352255941 z_k1_s1172
                                 - 0.324000540193767 z_k1_s1173
                                 - 0.247000522319518 z_k1_s1174
                                 - 0.638002436304969 z_k1_s1175
                                 - 0.543998976060194 z_k1_s1176
                                 - 0.61799949492984 z_k1_s1177 >= 0
 z_k1_s0000_constraint:          var_1 + z_k1_s0000 + 1.08477 w_k1_ADM_s000
                                 + 1.142 w_k1_AXP_s000 + 1.21645 w_k1_DIS_s000
                                 + 1.23234 w_k1_FDX_s000
                                 + 0.901213 w_k1_IBM_s000
                                 + 0.998464 w_k1_INTC_s000
                                 + 1.05336 w_k1_KO_s000 + 1.15594 w_k1_MAT_s000
                                 + 1.21812 w_k1_MSFT_s000
                                 + 1.18613 w_k1_PG_s000 >= 1
 z_k1_s0001_constraint:          var_1 + z_k1_s0001 + 1.05589 w_k1_ADM_s000
                                 + 1.09593 w_k1_AXP_s000
                                 + 1.15953 w_k1_DIS_s000
                                 + 1.04699 w_k1_FDX_s000 + 1.7749 w_k1_IBM_s000
                                 + 1.11313 w_k1_INTC_s000 + 1.4324 w_k1_KO_s000
                                 + 1.60834 w_k1_MAT_s000
                                 + 1.0707 w_k1_MSFT_s000 + 1.16317 w_k1_PG_s000
                                 >= 1
 z_k1_s0002_constraint:          var_1 + z_k1_s0002 + 0.710328 w_k1_ADM_s000
                                 + 0.720102 w_k1_AXP_s000
                                 + 0.758203 w_k1_DIS_s000
                                 + 0.796373 w_k1_FDX_s000
                                 + 1.15128 w_k1_IBM_s000
                                 + 0.828743 w_k1_INTC_s000
                                 + 0.968191 w_k1_KO_s000
                                 + 0.987136 w_k1_MAT_s000
                                 + 0.880551 w_k1_MSFT_s000
                                 + 0.845542 w_k1_PG_s000 >= 1
 z_k1_s0003_constraint:          var_1 + z_k1_s0003 + 0.688069 w_k1_ADM_s000
                                 + 0.636468 w_k1_AXP_s000
                                 + 0.824606 w_k1_DIS_s000
                                 + 0.855841 w_k1_FDX_s000
                                 + 0.616665 w_k1_IBM_s000
                                 + 0.772337 w_k1_INTC_s000
                                 + 0.786877 w_k1_KO_s000
                                 + 0.847147 w_k1_MAT_s000
                                 + 0.863231 w_k1_MSFT_s000
                                 + 0.933726 w_k1_PG_s000 >= 1
 z_k1_s0004_constraint:          var_1 + z_k1_s0004 + 1.0446 w_k1_ADM_s000
                                 + 1.07368 w_k1_AXP_s000
                                 + 1.03994 w_k1_DIS_s000
                                 + 1.11626 w_k1_FDX_s000
                                 + 0.64224 w_k1_IBM_s000
                                 + 0.946265 w_k1_INTC_s000
                                 + 0.774297 w_k1_KO_s000
                                 + 0.727418 w_k1_MAT_s000
                                 + 1.0469 w_k1_MSFT_s000 + 0.91865 w_k1_PG_s000
                                 >= 1
 z_k1_s0005_constraint:          var_1 + z_k1_s0005 + 1.16871 w_k1_ADM_s000
                                 + 1.05323 w_k1_AXP_s000
                                 + 0.994657 w_k1_DIS_s000
                                 + 1.03383 w_k1_FDX_s000
                                 + 1.01556 w_k1_IBM_s000
                                 + 0.928487 w_k1_INTC_s000
                                 + 1.05212 w_k1_KO_s000 + 1.67012 w_k1_MAT_s000
                                 + 1.00361 w_k1_MSFT_s000
                                 + 1.05446 w_k1_PG_s000 >= 1
 z_k1_s0006_constraint:          var_1 + z_k1_s0006 + 1.32577 w_k1_ADM_s000
                                 + 1.26528 w_k1_AXP_s000
                                 + 1.55458 w_k1_DIS_s000
                                 + 1.21825 w_k1_FDX_s000 + 1.19 w_k1_IBM_s000
                                 + 1.07465 w_k1_INTC_s000
                                 + 1.40794 w_k1_KO_s000 + 1.47751 w_k1_MAT_s000
                                 + 1.25318 w_k1_MSFT_s000 + 1.37 w_k1_PG_s000
                                 >= 1
 z_k1_s0007_constraint:          var_1 + z_k1_s0007 + 0.920179 w_k1_ADM_s000
                                 + 0.803867 w_k1_AXP_s000
                                 + 0.82075 w_k1_DIS_s000
                                 + 0.927486 w_k1_FDX_s000
                                 + 0.801996 w_k1_IBM_s000
                                 + 0.871893 w_k1_INTC_s000
                                 + 0.910507 w_k1_KO_s000
                                 + 1.04528 w_k1_MAT_s000
                                 + 0.98893 w_k1_MSFT_s000
                                 + 0.935004 w_k1_PG_s000 >= 1
 z_k1_s0010_constraint:          var_1 + z_k1_s0010 + 0.966935 w_k1_ADM_s001
                                 + 1.29798 w_k1_AXP_s001
                                 + 0.930721 w_k1_DIS_s001
                                 + 1.22985 w_k1_FDX_s001
                                 + 0.720133 w_k1_IBM_s001
                                 + 1.19236 w_k1_INTC_s001
                                 + 0.907164 w_k1_KO_s001
                                 + 1.02084 w_k1_MAT_s001
                                 + 1.05746 w_k1_MSFT_s001
                                 + 1.01221 w_k1_PG_s001 >= 1
 z_k1_s0011_constraint:          var_1 + z_k1_s0011 + 0.876755 w_k1_ADM_s001
                                 + 0.875155 w_k1_AXP_s001
                                 + 0.873703 w_k1_DIS_s001
                                 + 0.957745 w_k1_FDX_s001
                                 + 0.815221 w_k1_IBM_s001
                                 + 0.895595 w_k1_INTC_s001
                                 + 0.732445 w_k1_KO_s001
                                 + 0.951045 w_k1_MAT_s001
                                 + 0.949088 w_k1_MSFT_s001
                                 + 0.928778 w_k1_PG_s001 >= 1
 z_k1_s0012_constraint:          var_1 + z_k1_s0012 + 0.814007 w_k1_ADM_s001
                                 + 1.00213 w_k1_AXP_s001
                                 + 0.986972 w_k1_DIS_s001
                                 + 0.965106 w_k1_FDX_s001
                                 + 1.0565 w_k1_IBM_s001
                                 + 1.02968 w_k1_INTC_s001
                                 + 0.962167 w_k1_KO_s001
                                 + 1.20056 w_k1_MAT_s001
                                 + 1.00529 w_k1_MSFT_s001 + 1.0146 w_k1_PG_s001
                                 >= 1
 z_k1_s0013_constraint:          var_1 + z_k1_s0013 + 0.743682 w_k1_ADM_s001
                                 + 0.838699 w_k1_AXP_s001
                                 + 0.625176 w_k1_DIS_s001
                                 + 0.804691 w_k1_FDX_s001
                                 + 0.930349 w_k1_IBM_s001
                                 + 0.880906 w_k1_INTC_s001
                                 + 0.82402 w_k1_KO_s001
                                 + 0.985615 w_k1_MAT_s001
                                 + 0.804073 w_k1_MSFT_s001
                                 + 0.828451 w_k1_PG_s001 >= 1
 z_k1_s0014_constraint:          var_1 + z_k1_s0014 + 1.01547 w_k1_ADM_s001
                                 + 1.47968 w_k1_AXP_s001
                                 + 1.48671 w_k1_DIS_s001
                                 + 1.34432 w_k1_FDX_s001
                                 + 1.07777 w_k1_IBM_s001
                                 + 1.27335 w_k1_INTC_s001
                                 + 0.967678 w_k1_KO_s001
                                 + 1.32661 w_k1_MAT_s001
                                 + 1.26466 w_k1_MSFT_s001
                                 + 1.05545 w_k1_PG_s001 >= 1
 z_k1_s0015_constraint:          var_1 + z_k1_s0015 + 1.03257 w_k1_ADM_s001
                                 + 1.22749 w_k1_AXP_s001 + 1.3098 w_k1_DIS_s001
                                 + 1.09015 w_k1_FDX_s001
                                 + 1.60378 w_k1_IBM_s001
                                 + 1.21425 w_k1_INTC_s001
                                 + 1.31879 w_k1_KO_s001 + 1.57159 w_k1_MAT_s001
                                 + 1.11911 w_k1_MSFT_s001
                                 + 1.10804 w_k1_PG_s001 >= 1
 z_k1_s0016_constraint:          var_1 + z_k1_s0016 + 1.01723 w_k1_ADM_s001
                                 + 0.889795 w_k1_AXP_s001
                                 + 0.806682 w_k1_DIS_s001
                                 + 1.00497 w_k1_FDX_s001
                                 + 0.563056 w_k1_IBM_s001
                                 + 0.910284 w_k1_INTC_s001
                                 + 0.697668 w_k1_KO_s001
                                 + 0.715921 w_k1_MAT_s001
                                 + 0.938952 w_k1_MSFT_s001
                                 + 0.865525 w_k1_PG_s001 >= 1
 z_k1_s0017_constraint:          var_1 + z_k1_s0017 + 1.03545 w_k1_ADM_s001
                                 + 1.43875 w_k1_AXP_s001
                                 + 0.94341 w_k1_DIS_s001
                                 + 1.09853 w_k1_FDX_s001
                                 + 1.18179 w_k1_IBM_s001
                                 + 0.874949 w_k1_INTC_s001
                                 + 0.804787 w_k1_KO_s001
                                 + 1.42972 w_k1_MAT_s001
                                 + 1.02247 w_k1_MSFT_s001
                                 + 1.07641 w_k1_PG_s001 >= 1
 z_k1_s0020_constraint:          var_1 + z_k1_s0020 + 0.882624 w_k1_ADM_s002
                                 + 0.761249 w_k1_AXP_s002
                                 + 0.725846 w_k1_DIS_s002
                                 + 0.757415 w_k1_FDX_s002
                                 + 0.854564 w_k1_IBM_s002
                                 + 0.729865 w_k1_INTC_s002
                                 + 0.675994 w_k1_KO_s002
                                 + 0.425336 w_k1_MAT_s002
                                 + 0.679371 w_k1_MSFT_s002
                                 + 0.744177 w_k1_PG_s002 >= 1
 z_k1_s0021_constraint:          var_1 + z_k1_s0021 + 1.22729 w_k1_ADM_s002
                                 + 1.27162 w_k1_AXP_s002
                                 + 1.74602 w_k1_DIS_s002 + 1.1304 w_k1_FDX_s002
                                 + 1.38692 w_k1_IBM_s002
                                 + 1.1086 w_k1_INTC_s002 + 1.26436 w_k1_KO_s002
                                 + 1.10776 w_k1_MAT_s002
                                 + 1.16721 w_k1_MSFT_s002 + 1.1816 w_k1_PG_s002
                                 >= 1
 z_k1_s0022_constraint:          var_1 + z_k1_s0022 + 0.931835 w_k1_ADM_s002
                                 + 0.990575 w_k1_AXP_s002
                                 + 1.12064 w_k1_DIS_s002
                                 + 0.952064 w_k1_FDX_s002
                                 + 1.36253 w_k1_IBM_s002
                                 + 1.00364 w_k1_INTC_s002
                                 + 1.00532 w_k1_KO_s002
                                 + 0.999244 w_k1_MAT_s002
                                 + 1.04459 w_k1_MSFT_s002
                                 + 0.97459 w_k1_PG_s002 >= 1
 z_k1_s0023_constraint:          var_1 + z_k1_s0023 + 1.11822 w_k1_ADM_s002
                                 + 0.843156 w_k1_AXP_s002
                                 + 1.11131 w_k1_DIS_s002
                                 + 0.963885 w_k1_FDX_s002 + 0.81 w_k1_IBM_s002
                                 + 0.799001 w_k1_INTC_s002
                                 + 0.871167 w_k1_KO_s002
                                 + 0.74857 w_k1_MAT_s002
                                 + 0.981137 w_k1_MSFT_s002
                                 + 0.90636 w_k1_PG_s002 >= 1
 z_k1_s0024_constraint:          var_1 + z_k1_s0024 + 1.42932 w_k1_ADM_s002
                                 + 1.24916 w_k1_AXP_s002
                                 + 1.12497 w_k1_DIS_s002
                                 + 1.24375 w_k1_FDX_s002
                                 + 0.696664 w_k1_IBM_s002
                                 + 0.905797 w_k1_INTC_s002
                                 + 0.764324 w_k1_KO_s002
                                 + 0.560939 w_k1_MAT_s002
                                 + 0.945076 w_k1_MSFT_s002
                                 + 1.0047 w_k1_PG_s002 >= 1
 z_k1_s0025_constraint:          var_1 + z_k1_s0025 + 0.841038 w_k1_ADM_s002
                                 + 0.904264 w_k1_AXP_s002
                                 + 0.811405 w_k1_DIS_s002
                                 + 0.73168 w_k1_FDX_s002
                                 + 1.33704 w_k1_IBM_s002
                                 + 0.89982 w_k1_INTC_s002 + 1.1059 w_k1_KO_s002
                                 + 1.07326 w_k1_MAT_s002
                                 + 0.741579 w_k1_MSFT_s002
                                 + 0.937823 w_k1_PG_s002 >= 1
 z_k1_s0026_constraint:          var_1 + z_k1_s0026 + 1.42056 w_k1_ADM_s002
                                 + 1.62377 w_k1_AXP_s002
                                 + 1.92763 w_k1_DIS_s002
                                 + 1.54254 w_k1_FDX_s002
                                 + 1.14406 w_k1_IBM_s002
                                 + 1.40486 w_k1_INTC_s002
                                 + 1.12599 w_k1_KO_s002 + 1.10374 w_k1_MAT_s002
                                 + 1.45935 w_k1_MSFT_s002
                                 + 1.22264 w_k1_PG_s002 >= 1
 z_k1_s0027_constraint:          var_1 + z_k1_s0027 + 1.38749 w_k1_ADM_s002
                                 + 1.13016 w_k1_AXP_s002
                                 + 1.33132 w_k1_DIS_s002
                                 + 1.20292 w_k1_FDX_s002
                                 + 1.03394 w_k1_IBM_s002
                                 + 0.857275 w_k1_INTC_s002
                                 + 1.20802 w_k1_KO_s002 + 1.24826 w_k1_MAT_s002
                                 + 1.14628 w_k1_MSFT_s002 + 1.1259 w_k1_PG_s002
                                 >= 1
 z_k1_s0030_constraint:          var_1 + z_k1_s0030 + 1.16338 w_k1_ADM_s003
                                 + 0.928339 w_k1_AXP_s003
                                 + 1.06901 w_k1_DIS_s003
                                 + 1.04404 w_k1_FDX_s003
                                 + 0.803039 w_k1_IBM_s003
                                 + 1.04622 w_k1_INTC_s003
                                 + 1.03264 w_k1_KO_s003
                                 + 0.773113 w_k1_MAT_s003
                                 + 1.30586 w_k1_MSFT_s003
                                 + 1.11355 w_k1_PG_s003 >= 1
 z_k1_s0031_constraint:          var_1 + z_k1_s0031 + 0.908683 w_k1_ADM_s003
                                 + 1.15596 w_k1_AXP_s003
                                 + 0.972754 w_k1_DIS_s003
                                 + 0.981256 w_k1_FDX_s003
                                 + 0.758141 w_k1_IBM_s003
                                 + 0.884119 w_k1_INTC_s003
                                 + 0.771432 w_k1_KO_s003
                                 + 0.908783 w_k1_MAT_s003
                                 + 1.16094 w_k1_MSFT_s003
                                 + 0.909162 w_k1_PG_s003 >= 1
 z_k1_s0032_constraint:          var_1 + z_k1_s0032 + 1.02864 w_k1_ADM_s003
                                 + 0.910481 w_k1_AXP_s003
                                 + 1.06829 w_k1_DIS_s003
                                 + 0.903434 w_k1_FDX_s003
                                 + 1.08655 w_k1_IBM_s003
                                 + 1.10467 w_k1_INTC_s003
                                 + 1.21846 w_k1_KO_s003
                                 + 0.789052 w_k1_MAT_s003
                                 + 1.19187 w_k1_MSFT_s003 + 1.0924 w_k1_PG_s003
                                 >= 1
 z_k1_s0033_constraint:          var_1 + z_k1_s0033 + 0.760781 w_k1_ADM_s003
                                 + 0.863933 w_k1_AXP_s003
                                 + 0.755353 w_k1_DIS_s003
                                 + 0.765863 w_k1_FDX_s003
                                 + 0.914033 w_k1_IBM_s003
                                 + 0.771587 w_k1_INTC_s003
                                 + 0.832402 w_k1_KO_s003
                                 + 1.05735 w_k1_MAT_s003
                                 + 0.993747 w_k1_MSFT_s003
                                 + 0.782124 w_k1_PG_s003 >= 1
 z_k1_s0034_constraint:          var_1 + z_k1_s0034 + 0.672811 w_k1_ADM_s003
                                 + 0.692185 w_k1_AXP_s003
                                 + 0.69965 w_k1_DIS_s003
                                 + 0.733353 w_k1_FDX_s003
                                 + 0.94034 w_k1_IBM_s003
                                 + 0.806847 w_k1_INTC_s003
                                 + 0.911299 w_k1_KO_s003
                                 + 1.00804 w_k1_MAT_s003
                                 + 0.949919 w_k1_MSFT_s003
                                 + 0.808854 w_k1_PG_s003 >= 1
 z_k1_s0035_constraint:          var_1 + z_k1_s0035 + 1.11483 w_k1_ADM_s003
                                 + 1.39309 w_k1_AXP_s003
                                 + 1.37273 w_k1_DIS_s003
                                 + 1.18096 w_k1_FDX_s003
                                 + 1.65727 w_k1_IBM_s003
                                 + 0.981695 w_k1_INTC_s003
                                 + 1.58934 w_k1_KO_s003 + 1.76304 w_k1_MAT_s003
                                 + 1.34585 w_k1_MSFT_s003
                                 + 1.17529 w_k1_PG_s003 >= 1
 z_k1_s0036_constraint:          var_1 + z_k1_s0036 + 1.33506 w_k1_ADM_s003
                                 + 1.30611 w_k1_AXP_s003
                                 + 1.65342 w_k1_DIS_s003 + 1.2947 w_k1_FDX_s003
                                 + 0.867742 w_k1_IBM_s003
                                 + 0.975367 w_k1_INTC_s003
                                 + 1.12209 w_k1_KO_s003 + 1.02511 w_k1_MAT_s003
                                 + 1.55039 w_k1_MSFT_s003
                                 + 1.07816 w_k1_PG_s003 >= 1
 z_k1_s0037_constraint:          var_1 + z_k1_s0037 + 0.825327 w_k1_ADM_s003
                                 + 0.848914 w_k1_AXP_s003
                                 + 0.96173 w_k1_DIS_s003
                                 + 0.912131 w_k1_FDX_s003
                                 + 0.714562 w_k1_IBM_s003
                                 + 0.935667 w_k1_INTC_s003
                                 + 0.899069 w_k1_KO_s003
                                 + 0.832147 w_k1_MAT_s003
                                 + 1.04025 w_k1_MSFT_s003
                                 + 0.941186 w_k1_PG_s003 >= 1
 z_k1_s0040_constraint:          var_1 + z_k1_s0040 + 0.801326 w_k1_ADM_s004
                                 + 0.870781 w_k1_AXP_s004
                                 + 0.758913 w_k1_DIS_s004
                                 + 0.82487 w_k1_FDX_s004
                                 + 1.10365 w_k1_IBM_s004
                                 + 0.753559 w_k1_INTC_s004
                                 + 0.967683 w_k1_KO_s004 + 1.2402 w_k1_MAT_s004
                                 + 0.764228 w_k1_MSFT_s004
                                 + 0.908717 w_k1_PG_s004 >= 1
 z_k1_s0041_constraint:          var_1 + z_k1_s0041 + 1.1602 w_k1_ADM_s004
                                 + 1.32405 w_k1_AXP_s004
                                 + 1.11073 w_k1_DIS_s004
                                 + 1.14375 w_k1_FDX_s004
                                 + 1.15603 w_k1_IBM_s004
                                 + 1.20522 w_k1_INTC_s004
                                 + 0.966891 w_k1_KO_s004
                                 + 1.74573 w_k1_MAT_s004
                                 + 1.01096 w_k1_MSFT_s004
                                 + 1.13407 w_k1_PG_s004 >= 1
 z_k1_s0042_constraint:          var_1 + z_k1_s0042 + 1.25074 w_k1_ADM_s004
                                 + 1.61269 w_k1_AXP_s004
                                 + 1.13036 w_k1_DIS_s004
                                 + 1.16694 w_k1_FDX_s004
                                 + 1.16539 w_k1_IBM_s004
                                 + 1.07313 w_k1_INTC_s004
                                 + 0.921232 w_k1_KO_s004
                                 + 1.65198 w_k1_MAT_s004
                                 + 1.00432 w_k1_MSFT_s004
                                 + 1.11122 w_k1_PG_s004 >= 1
 z_k1_s0043_constraint:          var_1 + z_k1_s0043 + 1.43602 w_k1_ADM_s004
                                 + 1.62445 w_k1_AXP_s004
                                 + 1.46808 w_k1_DIS_s004
                                 + 1.48737 w_k1_FDX_s004
                                 + 1.06388 w_k1_IBM_s004
                                 + 1.19916 w_k1_INTC_s004
                                 + 1.08076 w_k1_KO_s004 + 1.37215 w_k1_MAT_s004
                                 + 1.25786 w_k1_MSFT_s004
                                 + 1.12714 w_k1_PG_s004 >= 1
 z_k1_s0044_constraint:          var_1 + z_k1_s0044 + 0.887682 w_k1_ADM_s004
                                 + 1.17274 w_k1_AXP_s004
                                 + 1.16358 w_k1_DIS_s004
                                 + 1.02312 w_k1_FDX_s004
                                 + 1.40303 w_k1_IBM_s004
                                 + 1.1654 w_k1_INTC_s004 + 1.09485 w_k1_KO_s004
                                 + 1.91836 w_k1_MAT_s004
                                 + 0.969525 w_k1_MSFT_s004
                                 + 0.969949 w_k1_PG_s004 >= 1
 z_k1_s0045_constraint:          var_1 + z_k1_s0045 + 1.18311 w_k1_ADM_s004
                                 + 1.19125 w_k1_AXP_s004
                                 + 1.05045 w_k1_DIS_s004
                                 + 1.15303 w_k1_FDX_s004
                                 + 0.889259 w_k1_IBM_s004
                                 + 1.13131 w_k1_INTC_s004
                                 + 0.915487 w_k1_KO_s004
                                 + 0.920431 w_k1_MAT_s004
                                 + 1.05569 w_k1_MSFT_s004 + 1.0192 w_k1_PG_s004
                                 >= 1
 z_k1_s0046_constraint:          var_1 + z_k1_s0046 + 0.832521 w_k1_ADM_s004
                                 + 0.862008 w_k1_AXP_s004
                                 + 0.809114 w_k1_DIS_s004
                                 + 0.939591 w_k1_FDX_s004
                                 + 0.78807 w_k1_IBM_s004
                                 + 1.2052 w_k1_INTC_s004
                                 + 0.863778 w_k1_KO_s004
                                 + 0.936313 w_k1_MAT_s004
                                 + 0.967133 w_k1_MSFT_s004
                                 + 0.903801 w_k1_PG_s004 >= 1
 z_k1_s0047_constraint:          var_1 + z_k1_s0047 + 1.10484 w_k1_ADM_s004
                                 + 1.24387 w_k1_AXP_s004
                                 + 1.67822 w_k1_DIS_s004
                                 + 1.19628 w_k1_FDX_s004
                                 + 1.35461 w_k1_IBM_s004
                                 + 1.26411 w_k1_INTC_s004
                                 + 1.36012 w_k1_KO_s004 + 1.08835 w_k1_MAT_s004
                                 + 1.27714 w_k1_MSFT_s004 + 1.1651 w_k1_PG_s004
                                 >= 1
 z_k1_s0050_constraint:          var_1 + z_k1_s0050 + 0.943216 w_k1_ADM_s005
                                 + 1.34919 w_k1_AXP_s005
                                 + 0.847422 w_k1_DIS_s005
                                 + 0.924441 w_k1_FDX_s005
                                 + 1.25526 w_k1_IBM_s005
                                 + 1.04265 w_k1_INTC_s005
                                 + 0.967808 w_k1_KO_s005
                                 + 1.65429 w_k1_MAT_s005
                                 + 0.819838 w_k1_MSFT_s005
                                 + 1.06395 w_k1_PG_s005 >= 1
 z_k1_s0051_constraint:          var_1 + z_k1_s0051 + 0.993765 w_k1_ADM_s005
                                 + 1.02822 w_k1_AXP_s005
                                 + 0.976361 w_k1_DIS_s005
                                 + 0.997123 w_k1_FDX_s005
                                 + 1.0165 w_k1_IBM_s005
                                 + 0.837304 w_k1_INTC_s005
                                 + 1.01278 w_k1_KO_s005 + 1.09358 w_k1_MAT_s005
                                 + 1.00056 w_k1_MSFT_s005
                                 + 1.05288 w_k1_PG_s005 >= 1
 z_k1_s0052_constraint:          var_1 + z_k1_s0052 + 0.742679 w_k1_ADM_s005
                                 + 0.978279 w_k1_AXP_s005
                                 + 0.726978 w_k1_DIS_s005
                                 + 0.979253 w_k1_FDX_s005
                                 + 0.662809 w_k1_IBM_s005
                                 + 1.14825 w_k1_INTC_s005
                                 + 0.65721 w_k1_KO_s005
                                 + 0.592966 w_k1_MAT_s005
                                 + 0.880255 w_k1_MSFT_s005
                                 + 0.826883 w_k1_PG_s005 >= 1
 z_k1_s0053_constraint:          var_1 + z_k1_s0053 + 0.998115 w_k1_ADM_s005
                                 + 0.950953 w_k1_AXP_s005
                                 + 0.643921 w_k1_DIS_s005
                                 + 0.68319 w_k1_FDX_s005
                                 + 1.12564 w_k1_IBM_s005
                                 + 0.701144 w_k1_INTC_s005
                                 + 0.730015 w_k1_KO_s005
                                 + 1.05721 w_k1_MAT_s005
                                 + 0.691699 w_k1_MSFT_s005
                                 + 0.847232 w_k1_PG_s005 >= 1
 z_k1_s0054_constraint:          var_1 + z_k1_s0054 + 0.973929 w_k1_ADM_s005
                                 + 1.32334 w_k1_AXP_s005
                                 + 1.37362 w_k1_DIS_s005
                                 + 1.09576 w_k1_FDX_s005
                                 + 1.33511 w_k1_IBM_s005
                                 + 1.20223 w_k1_INTC_s005
                                 + 1.15937 w_k1_KO_s005 + 1.6018 w_k1_MAT_s005
                                 + 0.980285 w_k1_MSFT_s005
                                 + 1.06005 w_k1_PG_s005 >= 1
 z_k1_s0055_constraint:          var_1 + z_k1_s0055 + 1.33614 w_k1_ADM_s005
                                 + 1.84128 w_k1_AXP_s005
                                 + 1.48014 w_k1_DIS_s005
                                 + 1.38689 w_k1_FDX_s005
                                 + 1.19615 w_k1_IBM_s005
                                 + 1.32034 w_k1_INTC_s005
                                 + 1.20832 w_k1_KO_s005 + 1.19678 w_k1_MAT_s005
                                 + 1.18767 w_k1_MSFT_s005
                                 + 1.21045 w_k1_PG_s005 >= 1
 z_k1_s0056_constraint:          var_1 + z_k1_s0056 + 1.00272 w_k1_ADM_s005
                                 + 0.756834 w_k1_AXP_s005
                                 + 0.873922 w_k1_DIS_s005
                                 + 0.824463 w_k1_FDX_s005
                                 + 1.02041 w_k1_IBM_s005
                                 + 0.690139 w_k1_INTC_s005
                                 + 0.919745 w_k1_KO_s005
                                 + 0.809997 w_k1_MAT_s005
                                 + 0.839522 w_k1_MSFT_s005
                                 + 0.837815 w_k1_PG_s005 >= 1
 z_k1_s0057_constraint:          var_1 + z_k1_s0057 + 0.840154 w_k1_ADM_s005
                                 + 0.794927 w_k1_AXP_s005
                                 + 0.713991 w_k1_DIS_s005
                                 + 0.796509 w_k1_FDX_s005
                                 + 0.780224 w_k1_IBM_s005
                                 + 0.811743 w_k1_INTC_s005
                                 + 0.755909 w_k1_KO_s005
                                 + 0.565709 w_k1_MAT_s005
                                 + 0.751838 w_k1_MSFT_s005
                                 + 0.853494 w_k1_PG_s005 >= 1
 z_k1_s0060_constraint:          var_1 + z_k1_s0060 + 1.00385 w_k1_ADM_s006
                                 + 0.639122 w_k1_AXP_s006
                                 + 0.77866 w_k1_DIS_s006
                                 + 0.821157 w_k1_FDX_s006
                                 + 0.887223 w_k1_IBM_s006
                                 + 0.62299 w_k1_INTC_s006
                                 + 0.95518 w_k1_KO_s006 + 0.75661 w_k1_MAT_s006
                                 + 0.79322 w_k1_MSFT_s006
                                 + 0.863826 w_k1_PG_s006 >= 1
 z_k1_s0061_constraint:          var_1 + z_k1_s0061 + 1.39298 w_k1_ADM_s006
                                 + 1.24837 w_k1_AXP_s006
                                 + 1.71041 w_k1_DIS_s006
                                 + 1.40435 w_k1_FDX_s006
                                 + 1.33969 w_k1_IBM_s006
                                 + 1.00273 w_k1_INTC_s006
                                 + 1.50513 w_k1_KO_s006 + 1.61694 w_k1_MAT_s006
                                 + 1.31058 w_k1_MSFT_s006
                                 + 1.21376 w_k1_PG_s006 >= 1
 z_k1_s0062_constraint:          var_1 + z_k1_s0062 + 1.11563 w_k1_ADM_s006
                                 + 1.08701 w_k1_AXP_s006 + 1.1618 w_k1_DIS_s006
                                 + 1.01323 w_k1_FDX_s006 + 1.2588 w_k1_IBM_s006
                                 + 0.976589 w_k1_INTC_s006
                                 + 1.22932 w_k1_KO_s006
                                 + 0.983223 w_k1_MAT_s006
                                 + 1.05407 w_k1_MSFT_s006
                                 + 1.08624 w_k1_PG_s006 >= 1
 z_k1_s0063_constraint:          var_1 + z_k1_s0063 + 1.02689 w_k1_ADM_s006
                                 + 0.904106 w_k1_AXP_s006
                                 + 1.05168 w_k1_DIS_s006
                                 + 0.992531 w_k1_FDX_s006
                                 + 0.991358 w_k1_IBM_s006
                                 + 0.888263 w_k1_INTC_s006
                                 + 0.952978 w_k1_KO_s006
                                 + 1.22621 w_k1_MAT_s006
                                 + 1.09418 w_k1_MSFT_s006
                                 + 0.968181 w_k1_PG_s006 >= 1
 z_k1_s0064_constraint:          var_1 + z_k1_s0064 + 0.825876 w_k1_ADM_s006
                                 + 0.78522 w_k1_AXP_s006
                                 + 0.898598 w_k1_DIS_s006
                                 + 1.11689 w_k1_FDX_s006
                                 + 0.662528 w_k1_IBM_s006
                                 + 0.976611 w_k1_INTC_s006
                                 + 0.864677 w_k1_KO_s006
                                 + 0.61527 w_k1_MAT_s006
                                 + 1.00493 w_k1_MSFT_s006
                                 + 0.93774 w_k1_PG_s006 >= 1
 z_k1_s0065_constraint:          var_1 + z_k1_s0065 + 1.24186 w_k1_ADM_s006
                                 + 1.2977 w_k1_AXP_s006 + 1.27581 w_k1_DIS_s006
                                 + 1.17233 w_k1_FDX_s006
                                 + 1.01371 w_k1_IBM_s006
                                 + 1.08423 w_k1_INTC_s006
                                 + 1.20235 w_k1_KO_s006 + 1.45024 w_k1_MAT_s006
                                 + 1.19013 w_k1_MSFT_s006
                                 + 1.04581 w_k1_PG_s006 >= 1
 z_k1_s0066_constraint:          var_1 + z_k1_s0066 + 1.20716 w_k1_ADM_s006
                                 + 1.00693 w_k1_AXP_s006
                                 + 0.99242 w_k1_DIS_s006
                                 + 1.06057 w_k1_FDX_s006
                                 + 0.968203 w_k1_IBM_s006
                                 + 0.783426 w_k1_INTC_s006
                                 + 1.07458 w_k1_KO_s006 + 1.19363 w_k1_MAT_s006
                                 + 1.00328 w_k1_MSFT_s006
                                 + 1.11916 w_k1_PG_s006 >= 1
 z_k1_s0067_constraint:          var_1 + z_k1_s0067 + 0.849232 w_k1_ADM_s006
                                 + 1.2766 w_k1_AXP_s006 + 1.76481 w_k1_DIS_s006
                                 + 1.08211 w_k1_FDX_s006
                                 + 1.67738 w_k1_IBM_s006
                                 + 1.15977 w_k1_INTC_s006
                                 + 1.73219 w_k1_KO_s006 + 1.9569 w_k1_MAT_s006
                                 + 1.22501 w_k1_MSFT_s006
                                 + 1.26154 w_k1_PG_s006 >= 1
 z_k1_s0070_constraint:          var_1 + z_k1_s0070 + 1.11473 w_k1_ADM_s007
                                 + 1.14726 w_k1_AXP_s007
                                 + 1.18169 w_k1_DIS_s007
                                 + 1.23799 w_k1_FDX_s007
                                 + 1.04204 w_k1_IBM_s007
                                 + 1.12712 w_k1_INTC_s007
                                 + 1.31444 w_k1_KO_s007 + 1.40861 w_k1_MAT_s007
                                 + 1.06142 w_k1_MSFT_s007
                                 + 1.17753 w_k1_PG_s007 >= 1
 z_k1_s0071_constraint:          var_1 + z_k1_s0071 + 1.30199 w_k1_ADM_s007
                                 + 1.32244 w_k1_AXP_s007
                                 + 1.21004 w_k1_DIS_s007
                                 + 1.22868 w_k1_FDX_s007
                                 + 1.30288 w_k1_IBM_s007
                                 + 1.40832 w_k1_INTC_s007
                                 + 1.34774 w_k1_KO_s007 + 1.21212 w_k1_MAT_s007
                                 + 1.32356 w_k1_MSFT_s007
                                 + 1.11341 w_k1_PG_s007 >= 1
 z_k1_s0072_constraint:          var_1 + z_k1_s0072 + 1.19451 w_k1_ADM_s007
                                 + 0.917297 w_k1_AXP_s007
                                 + 0.955224 w_k1_DIS_s007
                                 + 1.07612 w_k1_FDX_s007
                                 + 0.69281 w_k1_IBM_s007
                                 + 0.915976 w_k1_INTC_s007
                                 + 1.03032 w_k1_KO_s007
                                 + 0.883885 w_k1_MAT_s007
                                 + 1.13623 w_k1_MSFT_s007
                                 + 0.938604 w_k1_PG_s007 >= 1
 z_k1_s0073_constraint:          var_1 + z_k1_s0073 + 0.850157 w_k1_ADM_s007
                                 + 0.821676 w_k1_AXP_s007
                                 + 1.06939 w_k1_DIS_s007
                                 + 1.03531 w_k1_FDX_s007
                                 + 0.864394 w_k1_IBM_s007
                                 + 1.12526 w_k1_INTC_s007
                                 + 1.12007 w_k1_KO_s007
                                 + 0.928813 w_k1_MAT_s007
                                 + 1.1835 w_k1_MSFT_s007 + 1.00065 w_k1_PG_s007
                                 >= 1
 z_k1_s0074_constraint:          var_1 + z_k1_s0074 + 0.863171 w_k1_ADM_s007
                                 + 0.657153 w_k1_AXP_s007
                                 + 0.70689 w_k1_DIS_s007
                                 + 0.775233 w_k1_FDX_s007
                                 + 0.808882 w_k1_IBM_s007
                                 + 0.927921 w_k1_INTC_s007
                                 + 1.02246 w_k1_KO_s007
                                 + 0.982823 w_k1_MAT_s007
                                 + 0.935045 w_k1_MSFT_s007
                                 + 0.891942 w_k1_PG_s007 >= 1
 z_k1_s0075_constraint:          var_1 + z_k1_s0075 + 1.01852 w_k1_ADM_s007
                                 + 0.850321 w_k1_AXP_s007
                                 + 0.921288 w_k1_DIS_s007
                                 + 0.907874 w_k1_FDX_s007
                                 + 1.68595 w_k1_IBM_s007
                                 + 1.21793 w_k1_INTC_s007
                                 + 1.44594 w_k1_KO_s007 + 2.03786 w_k1_MAT_s007
                                 + 1.16364 w_k1_MSFT_s007
                                 + 1.11446 w_k1_PG_s007 >= 1
 z_k1_s0076_constraint:          var_1 + z_k1_s0076 + 0.881524 w_k1_ADM_s007
                                 + 0.852765 w_k1_AXP_s007
                                 + 0.926513 w_k1_DIS_s007
                                 + 0.930748 w_k1_FDX_s007
                                 + 1.16834 w_k1_IBM_s007
                                 + 1.12395 w_k1_INTC_s007
                                 + 1.23744 w_k1_KO_s007 + 1.29252 w_k1_MAT_s007
                                 + 1.03081 w_k1_MSFT_s007
                                 + 0.852989 w_k1_PG_s007 >= 1
 z_k1_s0077_constraint:          var_1 + z_k1_s0077 + 1.04937 w_k1_ADM_s007
                                 + 1.41664 w_k1_AXP_s007
                                 + 1.29105 w_k1_DIS_s007
                                 + 1.14136 w_k1_FDX_s007
                                 + 1.66205 w_k1_IBM_s007
                                 + 1.33836 w_k1_INTC_s007
                                 + 1.57135 w_k1_KO_s007 + 2.98833 w_k1_MAT_s007
                                 + 1.20382 w_k1_MSFT_s007
                                 + 1.26363 w_k1_PG_s007 >= 1
 z_k1_s0100_constraint:          var_1 + z_k1_s0100 + 0.643423 w_k1_ADM_s010
                                 + 0.727313 w_k1_AXP_s010
                                 + 0.606327 w_k1_DIS_s010
                                 + 0.740753 w_k1_FDX_s010
                                 + 0.780838 w_k1_IBM_s010
                                 + 0.803161 w_k1_INTC_s010
                                 + 0.708156 w_k1_KO_s010
                                 + 0.73892 w_k1_MAT_s010
                                 + 0.648631 w_k1_MSFT_s010
                                 + 0.834548 w_k1_PG_s010 >= 1
 z_k1_s0101_constraint:          var_1 + z_k1_s0101 + 1.34064 w_k1_ADM_s010
                                 + 0.946063 w_k1_AXP_s010
                                 + 1.01498 w_k1_DIS_s010
                                 + 0.978969 w_k1_FDX_s010
                                 + 1.0337 w_k1_IBM_s010
                                 + 1.13144 w_k1_INTC_s010
                                 + 1.18039 w_k1_KO_s010
                                 + 0.977507 w_k1_MAT_s010
                                 + 0.872908 w_k1_MSFT_s010
                                 + 1.02747 w_k1_PG_s010 >= 1
 z_k1_s0102_constraint:          var_1 + z_k1_s0102 + 0.951559 w_k1_ADM_s010
                                 + 0.826507 w_k1_AXP_s010
                                 + 0.770774 w_k1_DIS_s010
                                 + 0.793021 w_k1_FDX_s010
                                 + 1.08182 w_k1_IBM_s010
                                 + 0.893067 w_k1_INTC_s010
                                 + 0.948451 w_k1_KO_s010
                                 + 0.992243 w_k1_MAT_s010
                                 + 0.743777 w_k1_MSFT_s010
                                 + 0.953971 w_k1_PG_s010 >= 1
 z_k1_s0103_constraint:          var_1 + z_k1_s0103 + 1.03832 w_k1_ADM_s010
                                 + 1.51795 w_k1_AXP_s010
                                 + 1.29902 w_k1_DIS_s010
                                 + 1.22197 w_k1_FDX_s010 + 1.3965 w_k1_IBM_s010
                                 + 1.2659 w_k1_INTC_s010 + 1.38661 w_k1_KO_s010
                                 + 1.45812 w_k1_MAT_s010
                                 + 1.09625 w_k1_MSFT_s010
                                 + 1.09735 w_k1_PG_s010 >= 1
 z_k1_s0104_constraint:          var_1 + z_k1_s0104 + 0.746956 w_k1_ADM_s010
                                 + 1.06129 w_k1_AXP_s010
                                 + 0.990225 w_k1_DIS_s010
                                 + 0.957201 w_k1_FDX_s010
                                 + 1.29025 w_k1_IBM_s010
                                 + 1.15172 w_k1_INTC_s010
                                 + 1.02069 w_k1_KO_s010 + 1.37731 w_k1_MAT_s010
                                 + 0.879617 w_k1_MSFT_s010
                                 + 1.01923 w_k1_PG_s010 >= 1
 z_k1_s0105_constraint:          var_1 + z_k1_s0105 + 0.844457 w_k1_ADM_s010
                                 + 0.747289 w_k1_AXP_s010
                                 + 0.805317 w_k1_DIS_s010
                                 + 0.945266 w_k1_FDX_s010
                                 + 0.755415 w_k1_IBM_s010
                                 + 0.906832 w_k1_INTC_s010
                                 + 0.813474 w_k1_KO_s010
                                 + 0.958814 w_k1_MAT_s010
                                 + 0.901883 w_k1_MSFT_s010
                                 + 0.999441 w_k1_PG_s010 >= 1
 z_k1_s0106_constraint:          var_1 + z_k1_s0106 + 1.09266 w_k1_ADM_s010
                                 + 1.01252 w_k1_AXP_s010
                                 + 1.26755 w_k1_DIS_s010
                                 + 1.19531 w_k1_FDX_s010
                                 + 0.931662 w_k1_IBM_s010
                                 + 1.28732 w_k1_INTC_s010
                                 + 1.11936 w_k1_KO_s010 + 0.81913 w_k1_MAT_s010
                                 + 1.06148 w_k1_MSFT_s010
                                 + 0.911683 w_k1_PG_s010 >= 1
 z_k1_s0107_constraint:          var_1 + z_k1_s0107 + 0.934026 w_k1_ADM_s010
                                 + 1.03569 w_k1_AXP_s010
                                 + 0.885404 w_k1_DIS_s010
                                 + 1.02097 w_k1_FDX_s010
                                 + 0.71633 w_k1_IBM_s010
                                 + 1.22365 w_k1_INTC_s010
                                 + 0.854401 w_k1_KO_s010
                                 + 0.677654 w_k1_MAT_s010
                                 + 1.00225 w_k1_MSFT_s010
                                 + 0.941837 w_k1_PG_s010 >= 1
 z_k1_s0110_constraint:          var_1 + z_k1_s0110 + 0.931641 w_k1_ADM_s011
                                 + 0.913073 w_k1_AXP_s011
                                 + 0.894034 w_k1_DIS_s011
                                 + 0.923583 w_k1_FDX_s011
                                 + 1.14977 w_k1_IBM_s011
                                 + 0.911862 w_k1_INTC_s011
                                 + 0.882348 w_k1_KO_s011
                                 + 1.30382 w_k1_MAT_s011
                                 + 0.829005 w_k1_MSFT_s011
                                 + 1.01139 w_k1_PG_s011 >= 1
 z_k1_s0111_constraint:          var_1 + z_k1_s0111 + 0.761637 w_k1_ADM_s011
                                 + 0.780453 w_k1_AXP_s011
                                 + 0.75834 w_k1_DIS_s011
                                 + 0.747518 w_k1_FDX_s011
                                 + 0.822983 w_k1_IBM_s011
                                 + 0.778059 w_k1_INTC_s011
                                 + 0.753757 w_k1_KO_s011
                                 + 0.912452 w_k1_MAT_s011
                                 + 0.734406 w_k1_MSFT_s011
                                 + 0.944417 w_k1_PG_s011 >= 1
 z_k1_s0112_constraint:          var_1 + z_k1_s0112 + 1.01265 w_k1_ADM_s011
                                 + 1.51006 w_k1_AXP_s011
                                 + 1.17151 w_k1_DIS_s011
                                 + 1.04677 w_k1_FDX_s011
                                 + 1.42757 w_k1_IBM_s011
                                 + 1.19896 w_k1_INTC_s011
                                 + 1.16932 w_k1_KO_s011 + 1.84991 w_k1_MAT_s011
                                 + 0.988533 w_k1_MSFT_s011
                                 + 1.19543 w_k1_PG_s011 >= 1
 z_k1_s0113_constraint:          var_1 + z_k1_s0113 + 0.882233 w_k1_ADM_s011
                                 + 1.00361 w_k1_AXP_s011
                                 + 0.757649 w_k1_DIS_s011
                                 + 0.955949 w_k1_FDX_s011
                                 + 0.847234 w_k1_IBM_s011
                                 + 0.964732 w_k1_INTC_s011
                                 + 0.770903 w_k1_KO_s011
                                 + 0.792886 w_k1_MAT_s011
                                 + 0.84291 w_k1_MSFT_s011
                                 + 0.874453 w_k1_PG_s011 >= 1
 z_k1_s0114_constraint:          var_1 + z_k1_s0114 + 1.10819 w_k1_ADM_s011
                                 + 1.76933 w_k1_AXP_s011
                                 + 1.54566 w_k1_DIS_s011
                                 + 1.46704 w_k1_FDX_s011
                                 + 1.26323 w_k1_IBM_s011
                                 + 1.24373 w_k1_INTC_s011 + 1.1945 w_k1_KO_s011
                                 + 1.93825 w_k1_MAT_s011
                                 + 1.21236 w_k1_MSFT_s011
                                 + 1.41295 w_k1_PG_s011 >= 1
 z_k1_s0115_constraint:          var_1 + z_k1_s0115 + 1.56118 w_k1_ADM_s011
                                 + 1.39488 w_k1_AXP_s011
                                 + 1.05149 w_k1_DIS_s011
                                 + 1.28314 w_k1_FDX_s011
                                 + 1.03198 w_k1_IBM_s011
                                 + 1.13489 w_k1_INTC_s011
                                 + 0.834604 w_k1_KO_s011
                                 + 1.04339 w_k1_MAT_s011
                                 + 1.05644 w_k1_MSFT_s011
                                 + 1.16206 w_k1_PG_s011 >= 1
 z_k1_s0116_constraint:          var_1 + z_k1_s0116 + 1.01845 w_k1_ADM_s011
                                 + 1.05882 w_k1_AXP_s011
                                 + 1.18385 w_k1_DIS_s011
                                 + 1.11443 w_k1_FDX_s011
                                 + 1.04529 w_k1_IBM_s011
                                 + 0.880543 w_k1_INTC_s011
                                 + 0.922171 w_k1_KO_s011
                                 + 1.34037 w_k1_MAT_s011
                                 + 1.03382 w_k1_MSFT_s011
                                 + 0.980444 w_k1_PG_s011 >= 1
 z_k1_s0117_constraint:          var_1 + z_k1_s0117 + 0.892543 w_k1_ADM_s011
                                 + 0.973267 w_k1_AXP_s011
                                 + 1.58272 w_k1_DIS_s011
                                 + 1.17067 w_k1_FDX_s011
                                 + 1.15517 w_k1_IBM_s011
                                 + 1.19077 w_k1_INTC_s011
                                 + 1.09078 w_k1_KO_s011
                                 + 0.848397 w_k1_MAT_s011
                                 + 1.20213 w_k1_MSFT_s011
                                 + 1.00397 w_k1_PG_s011 >= 1
 z_k1_s0120_constraint:          var_1 + z_k1_s0120 + 1.35455 w_k1_ADM_s012
                                 + 1.16402 w_k1_AXP_s012
                                 + 1.40018 w_k1_DIS_s012
                                 + 1.26697 w_k1_FDX_s012
                                 + 0.922926 w_k1_IBM_s012
                                 + 0.790097 w_k1_INTC_s012
                                 + 1.06351 w_k1_KO_s012 + 1.08517 w_k1_MAT_s012
                                 + 1.09988 w_k1_MSFT_s012
                                 + 1.05231 w_k1_PG_s012 >= 1
 z_k1_s0121_constraint:          var_1 + z_k1_s0121 + 1.36705 w_k1_ADM_s012
                                 + 1.07462 w_k1_AXP_s012
                                 + 1.04329 w_k1_DIS_s012
                                 + 1.18278 w_k1_FDX_s012
                                 + 0.962839 w_k1_IBM_s012
                                 + 0.910216 w_k1_INTC_s012
                                 + 1.02287 w_k1_KO_s012
                                 + 0.914294 w_k1_MAT_s012
                                 + 1.1795 w_k1_MSFT_s012 + 1.03036 w_k1_PG_s012
                                 >= 1
 z_k1_s0122_constraint:          var_1 + z_k1_s0122 + 0.78677 w_k1_ADM_s012
                                 + 0.653659 w_k1_AXP_s012
                                 + 0.919254 w_k1_DIS_s012
                                 + 0.968298 w_k1_FDX_s012
                                 + 0.78089 w_k1_IBM_s012
                                 + 0.761082 w_k1_INTC_s012
                                 + 0.923139 w_k1_KO_s012
                                 + 0.753748 w_k1_MAT_s012
                                 + 0.938413 w_k1_MSFT_s012
                                 + 0.879059 w_k1_PG_s012 >= 1
 z_k1_s0123_constraint:          var_1 + z_k1_s0123 + 1.02383 w_k1_ADM_s012
                                 + 1.08201 w_k1_AXP_s012
                                 + 1.12345 w_k1_DIS_s012
                                 + 1.03861 w_k1_FDX_s012
                                 + 1.25034 w_k1_IBM_s012
                                 + 1.00705 w_k1_INTC_s012
                                 + 1.30105 w_k1_KO_s012 + 1.55982 w_k1_MAT_s012
                                 + 1.1902 w_k1_MSFT_s012 + 1.06661 w_k1_PG_s012
                                 >= 1
 z_k1_s0124_constraint:          var_1 + z_k1_s0124 + 1.28243 w_k1_ADM_s012
                                 + 1.26299 w_k1_AXP_s012
                                 + 1.48408 w_k1_DIS_s012
                                 + 1.15366 w_k1_FDX_s012
                                 + 1.52517 w_k1_IBM_s012
                                 + 1.05402 w_k1_INTC_s012
                                 + 1.33192 w_k1_KO_s012 + 1.26906 w_k1_MAT_s012
                                 + 1.28277 w_k1_MSFT_s012
                                 + 1.03801 w_k1_PG_s012 >= 1
 z_k1_s0125_constraint:          var_1 + z_k1_s0125 + 0.943394 w_k1_ADM_s012
                                 + 0.788574 w_k1_AXP_s012
                                 + 0.856865 w_k1_DIS_s012
                                 + 0.890685 w_k1_FDX_s012
                                 + 0.977374 w_k1_IBM_s012
                                 + 0.925159 w_k1_INTC_s012
                                 + 0.968973 w_k1_KO_s012
                                 + 0.887815 w_k1_MAT_s012
                                 + 0.987127 w_k1_MSFT_s012
                                 + 1.00893 w_k1_PG_s012 >= 1
 z_k1_s0126_constraint:          var_1 + z_k1_s0126 + 0.896452 w_k1_ADM_s012
                                 + 0.53904 w_k1_AXP_s012
                                 + 0.539908 w_k1_DIS_s012
                                 + 0.707833 w_k1_FDX_s012
                                 + 0.723375 w_k1_IBM_s012
                                 + 0.710502 w_k1_INTC_s012
                                 + 0.702639 w_k1_KO_s012
                                 + 0.65813 w_k1_MAT_s012
                                 + 0.798425 w_k1_MSFT_s012
                                 + 0.83738 w_k1_PG_s012 >= 1
 z_k1_s0127_constraint:          var_1 + z_k1_s0127 + 0.664978 w_k1_ADM_s012
                                 + 0.975553 w_k1_AXP_s012
                                 + 0.659162 w_k1_DIS_s012
                                 + 0.759177 w_k1_FDX_s012
                                 + 1.1614 w_k1_IBM_s012
                                 + 0.676277 w_k1_INTC_s012
                                 + 0.856853 w_k1_KO_s012 + 1.3612 w_k1_MAT_s012
                                 + 0.674064 w_k1_MSFT_s012
                                 + 0.814066 w_k1_PG_s012 >= 1
 z_k1_s0130_constraint:          var_1 + z_k1_s0130 + 0.952121 w_k1_ADM_s013
                                 + 1.27713 w_k1_AXP_s013
                                 + 0.768428 w_k1_DIS_s013
                                 + 1.08352 w_k1_FDX_s013
                                 + 0.755515 w_k1_IBM_s013
                                 + 0.985484 w_k1_INTC_s013
                                 + 0.713152 w_k1_KO_s013
                                 + 0.791714 w_k1_MAT_s013
                                 + 0.790597 w_k1_MSFT_s013
                                 + 0.88197 w_k1_PG_s013 >= 1
 z_k1_s0131_constraint:          var_1 + z_k1_s0131 + 1.07631 w_k1_ADM_s013
                                 + 1.17435 w_k1_AXP_s013 + 1.2245 w_k1_DIS_s013
                                 + 1.34839 w_k1_FDX_s013
                                 + 0.688453 w_k1_IBM_s013
                                 + 1.25769 w_k1_INTC_s013
                                 + 0.967446 w_k1_KO_s013
                                 + 0.700135 w_k1_MAT_s013
                                 + 1.28023 w_k1_MSFT_s013
                                 + 0.966368 w_k1_PG_s013 >= 1
 z_k1_s0132_constraint:          var_1 + z_k1_s0132 + 0.947011 w_k1_ADM_s013
                                 + 1.20672 w_k1_AXP_s013
                                 + 1.02866 w_k1_DIS_s013
                                 + 0.979706 w_k1_FDX_s013
                                 + 1.20956 w_k1_IBM_s013
                                 + 0.874631 w_k1_INTC_s013
                                 + 0.993303 w_k1_KO_s013
                                 + 1.32959 w_k1_MAT_s013
                                 + 0.966215 w_k1_MSFT_s013
                                 + 1.01563 w_k1_PG_s013 >= 1
 z_k1_s0133_constraint:          var_1 + z_k1_s0133 + 0.860784 w_k1_ADM_s013
                                 + 0.731231 w_k1_AXP_s013
                                 + 0.690152 w_k1_DIS_s013
                                 + 0.839799 w_k1_FDX_s013
                                 + 0.585385 w_k1_IBM_s013
                                 + 1.01742 w_k1_INTC_s013
                                 + 0.795701 w_k1_KO_s013
                                 + 0.490294 w_k1_MAT_s013
                                 + 0.964612 w_k1_MSFT_s013
                                 + 0.789082 w_k1_PG_s013 >= 1
 z_k1_s0134_constraint:          var_1 + z_k1_s0134 + 0.695272 w_k1_ADM_s013
                                 + 1.10256 w_k1_AXP_s013
                                 + 1.54444 w_k1_DIS_s013
                                 + 1.23086 w_k1_FDX_s013
                                 + 1.06584 w_k1_IBM_s013
                                 + 1.24264 w_k1_INTC_s013
                                 + 1.42371 w_k1_KO_s013 + 1.0385 w_k1_MAT_s013
                                 + 1.30744 w_k1_MSFT_s013
                                 + 1.13904 w_k1_PG_s013 >= 1
 z_k1_s0135_constraint:          var_1 + z_k1_s0135 + 1.00509 w_k1_ADM_s013
                                 + 1.33744 w_k1_AXP_s013
                                 + 1.40623 w_k1_DIS_s013
                                 + 1.17106 w_k1_FDX_s013 + 2.0551 w_k1_IBM_s013
                                 + 1.11242 w_k1_INTC_s013
                                 + 1.45939 w_k1_KO_s013 + 2.1835 w_k1_MAT_s013
                                 + 1.21626 w_k1_MSFT_s013
                                 + 1.20832 w_k1_PG_s013 >= 1
 z_k1_s0136_constraint:          var_1 + z_k1_s0136 + 0.630208 w_k1_ADM_s013
                                 + 0.88791 w_k1_AXP_s013
                                 + 0.694302 w_k1_DIS_s013
                                 + 0.812406 w_k1_FDX_s013
                                 + 0.978772 w_k1_IBM_s013
                                 + 0.983767 w_k1_INTC_s013
                                 + 0.861956 w_k1_KO_s013
                                 + 1.14943 w_k1_MAT_s013
                                 + 0.898348 w_k1_MSFT_s013
                                 + 0.801809 w_k1_PG_s013 >= 1
 z_k1_s0137_constraint:          var_1 + z_k1_s0137 + 0.87367 w_k1_ADM_s013
                                 + 0.766488 w_k1_AXP_s013
                                 + 0.936752 w_k1_DIS_s013
                                 + 0.832383 w_k1_FDX_s013
                                 + 1.21984 w_k1_IBM_s013
                                 + 0.823501 w_k1_INTC_s013
                                 + 1.06303 w_k1_KO_s013 + 1.05954 w_k1_MAT_s013
                                 + 0.89589 w_k1_MSFT_s013
                                 + 0.761326 w_k1_PG_s013 >= 1
 z_k1_s0140_constraint:          var_1 + z_k1_s0140 + 1.06114 w_k1_ADM_s014
                                 + 0.974755 w_k1_AXP_s014
                                 + 0.787435 w_k1_DIS_s014
                                 + 0.920626 w_k1_FDX_s014
                                 + 0.770355 w_k1_IBM_s014
                                 + 0.888646 w_k1_INTC_s014
                                 + 0.734376 w_k1_KO_s014
                                 + 0.755217 w_k1_MAT_s014
                                 + 0.851465 w_k1_MSFT_s014
                                 + 0.868401 w_k1_PG_s014 >= 1
 z_k1_s0141_constraint:          var_1 + z_k1_s0141 + 1.18313 w_k1_ADM_s014
                                 + 1.03634 w_k1_AXP_s014
                                 + 0.930575 w_k1_DIS_s014
                                 + 1.10549 w_k1_FDX_s014
                                 + 0.976814 w_k1_IBM_s014
                                 + 1.08968 w_k1_INTC_s014
                                 + 0.995158 w_k1_KO_s014
                                 + 0.941687 w_k1_MAT_s014
                                 + 1.02079 w_k1_MSFT_s014
                                 + 1.10772 w_k1_PG_s014 >= 1
 z_k1_s0142_constraint:          var_1 + z_k1_s0142 + 0.946833 w_k1_ADM_s014
                                 + 0.681791 w_k1_AXP_s014
                                 + 0.682896 w_k1_DIS_s014
                                 + 0.809446 w_k1_FDX_s014
                                 + 0.750889 w_k1_IBM_s014
                                 + 0.886881 w_k1_INTC_s014
                                 + 0.720456 w_k1_KO_s014
                                 + 0.685322 w_k1_MAT_s014
                                 + 0.784936 w_k1_MSFT_s014
                                 + 0.874679 w_k1_PG_s014 >= 1
 z_k1_s0143_constraint:          var_1 + z_k1_s0143 + 1.16531 w_k1_ADM_s014
                                 + 1.31618 w_k1_AXP_s014
                                 + 1.59102 w_k1_DIS_s014 + 1.2697 w_k1_FDX_s014
                                 + 1.24341 w_k1_IBM_s014
                                 + 1.32866 w_k1_INTC_s014
                                 + 1.24451 w_k1_KO_s014 + 1.02868 w_k1_MAT_s014
                                 + 1.30157 w_k1_MSFT_s014 + 1.1492 w_k1_PG_s014
                                 >= 1
 z_k1_s0144_constraint:          var_1 + z_k1_s0144 + 1.52855 w_k1_ADM_s014
                                 + 1.76449 w_k1_AXP_s014
                                 + 1.62241 w_k1_DIS_s014
                                 + 1.46785 w_k1_FDX_s014 + 1.2911 w_k1_IBM_s014
                                 + 1.32276 w_k1_INTC_s014
                                 + 1.16762 w_k1_KO_s014 + 1.69794 w_k1_MAT_s014
                                 + 1.40483 w_k1_MSFT_s014
                                 + 1.36603 w_k1_PG_s014 >= 1
 z_k1_s0145_constraint:          var_1 + z_k1_s0145 + 0.896665 w_k1_ADM_s014
                                 + 0.873927 w_k1_AXP_s014
                                 + 0.770556 w_k1_DIS_s014
                                 + 0.790562 w_k1_FDX_s014
                                 + 0.998094 w_k1_IBM_s014
                                 + 0.755869 w_k1_INTC_s014
                                 + 0.769461 w_k1_KO_s014
                                 + 1.26318 w_k1_MAT_s014
                                 + 0.801176 w_k1_MSFT_s014
                                 + 0.897041 w_k1_PG_s014 >= 1
 z_k1_s0146_constraint:          var_1 + z_k1_s0146 + 1.31274 w_k1_ADM_s014
                                 + 0.962879 w_k1_AXP_s014
                                 + 1.27302 w_k1_DIS_s014
                                 + 1.08488 w_k1_FDX_s014
                                 + 0.901595 w_k1_IBM_s014
                                 + 1.18151 w_k1_INTC_s014
                                 + 1.01399 w_k1_KO_s014
                                 + 0.548007 w_k1_MAT_s014
                                 + 1.13262 w_k1_MSFT_s014
                                 + 1.00621 w_k1_PG_s014 >= 1
 z_k1_s0147_constraint:          var_1 + z_k1_s0147 + 1.32054 w_k1_ADM_s014
                                 + 1.26025 w_k1_AXP_s014
                                 + 1.20817 w_k1_DIS_s014
                                 + 1.11755 w_k1_FDX_s014
                                 + 1.30974 w_k1_IBM_s014
                                 + 1.11883 w_k1_INTC_s014
                                 + 1.19874 w_k1_KO_s014 + 1.73781 w_k1_MAT_s014
                                 + 1.14671 w_k1_MSFT_s014 + 1.0697 w_k1_PG_s014
                                 >= 1
 z_k1_s0150_constraint:          var_1 + z_k1_s0150 + 0.774522 w_k1_ADM_s015
                                 + 0.933727 w_k1_AXP_s015
                                 + 1.20131 w_k1_DIS_s015 + 1.0582 w_k1_FDX_s015
                                 + 0.769556 w_k1_IBM_s015
                                 + 1.17594 w_k1_INTC_s015
                                 + 0.938265 w_k1_KO_s015
                                 + 0.615652 w_k1_MAT_s015
                                 + 0.983633 w_k1_MSFT_s015
                                 + 0.994884 w_k1_PG_s015 >= 1
 z_k1_s0151_constraint:          var_1 + z_k1_s0151 + 0.557794 w_k1_ADM_s015
                                 + 0.745657 w_k1_AXP_s015
                                 + 0.572605 w_k1_DIS_s015
                                 + 0.798524 w_k1_FDX_s015
                                 + 0.611117 w_k1_IBM_s015
                                 + 0.942481 w_k1_INTC_s015
                                 + 0.705347 w_k1_KO_s015
                                 + 0.734267 w_k1_MAT_s015
                                 + 0.728375 w_k1_MSFT_s015
                                 + 0.747376 w_k1_PG_s015 >= 1
 z_k1_s0152_constraint:          var_1 + z_k1_s0152 + 0.705869 w_k1_ADM_s015
                                 + 0.856665 w_k1_AXP_s015
                                 + 0.685747 w_k1_DIS_s015
                                 + 0.851674 w_k1_FDX_s015
                                 + 0.831087 w_k1_IBM_s015
                                 + 0.850394 w_k1_INTC_s015
                                 + 0.828398 w_k1_KO_s015
                                 + 0.682533 w_k1_MAT_s015
                                 + 0.759298 w_k1_MSFT_s015
                                 + 0.918214 w_k1_PG_s015 >= 1
 z_k1_s0153_constraint:          var_1 + z_k1_s0153 + 0.644942 w_k1_ADM_s015
                                 + 1.15275 w_k1_AXP_s015
                                 + 0.955633 w_k1_DIS_s015
                                 + 0.906628 w_k1_FDX_s015
                                 + 1.21108 w_k1_IBM_s015
                                 + 1.02088 w_k1_INTC_s015
                                 + 0.981395 w_k1_KO_s015
                                 + 1.67842 w_k1_MAT_s015
                                 + 0.957161 w_k1_MSFT_s015
                                 + 0.937361 w_k1_PG_s015 >= 1
 z_k1_s0154_constraint:          var_1 + z_k1_s0154 + 1.02934 w_k1_ADM_s015
                                 + 1.702 w_k1_AXP_s015 + 1.58317 w_k1_DIS_s015
                                 + 1.30205 w_k1_FDX_s015
                                 + 1.71395 w_k1_IBM_s015
                                 + 1.19551 w_k1_INTC_s015
                                 + 1.44917 w_k1_KO_s015 + 1.74802 w_k1_MAT_s015
                                 + 1.12437 w_k1_MSFT_s015
                                 + 1.26337 w_k1_PG_s015 >= 1
 z_k1_s0155_constraint:          var_1 + z_k1_s0155 + 0.540457 w_k1_ADM_s015
                                 + 0.670838 w_k1_AXP_s015
                                 + 0.815234 w_k1_DIS_s015
                                 + 0.845422 w_k1_FDX_s015
                                 + 0.724091 w_k1_IBM_s015
                                 + 0.758709 w_k1_INTC_s015
                                 + 0.70979 w_k1_KO_s015
                                 + 0.726054 w_k1_MAT_s015
                                 + 0.871204 w_k1_MSFT_s015
                                 + 0.831164 w_k1_PG_s015 >= 1
 z_k1_s0156_constraint:          var_1 + z_k1_s0156 + 1.05446 w_k1_ADM_s015
                                 + 1.20819 w_k1_AXP_s015
                                 + 1.23693 w_k1_DIS_s015
                                 + 1.19114 w_k1_FDX_s015
                                 + 1.11645 w_k1_IBM_s015
                                 + 1.07775 w_k1_INTC_s015
                                 + 1.19983 w_k1_KO_s015 + 1.3441 w_k1_MAT_s015
                                 + 1.10799 w_k1_MSFT_s015
                                 + 1.17832 w_k1_PG_s015 >= 1
 z_k1_s0157_constraint:          var_1 + z_k1_s0157 + 0.702292 w_k1_ADM_s015
                                 + 0.871565 w_k1_AXP_s015
                                 + 0.849656 w_k1_DIS_s015
                                 + 0.837492 w_k1_FDX_s015
                                 + 1.17576 w_k1_IBM_s015
                                 + 0.954953 w_k1_INTC_s015
                                 + 1.1518 w_k1_KO_s015 + 1.09531 w_k1_MAT_s015
                                 + 0.860385 w_k1_MSFT_s015
                                 + 0.94443 w_k1_PG_s015 >= 1
 z_k1_s0160_constraint:          var_1 + z_k1_s0160 + 0.988475 w_k1_ADM_s016
                                 + 0.800252 w_k1_AXP_s016
                                 + 0.736786 w_k1_DIS_s016
                                 + 0.706331 w_k1_FDX_s016
                                 + 1.04014 w_k1_IBM_s016
                                 + 0.786231 w_k1_INTC_s016
                                 + 0.942888 w_k1_KO_s016
                                 + 1.15436 w_k1_MAT_s016
                                 + 0.774925 w_k1_MSFT_s016
                                 + 0.991134 w_k1_PG_s016 >= 1
 z_k1_s0161_constraint:          var_1 + z_k1_s0161 + 1.58547 w_k1_ADM_s016
                                 + 1.37222 w_k1_AXP_s016
                                 + 1.27612 w_k1_DIS_s016
                                 + 1.21597 w_k1_FDX_s016
                                 + 0.828875 w_k1_IBM_s016
                                 + 1.07696 w_k1_INTC_s016
                                 + 1.05996 w_k1_KO_s016
                                 + 0.788538 w_k1_MAT_s016
                                 + 1.10422 w_k1_MSFT_s016
                                 + 1.06063 w_k1_PG_s016 >= 1
 z_k1_s0162_constraint:          var_1 + z_k1_s0162 + 0.979553 w_k1_ADM_s016
                                 + 1.06804 w_k1_AXP_s016
                                 + 0.942218 w_k1_DIS_s016
                                 + 0.87431 w_k1_FDX_s016
                                 + 1.08377 w_k1_IBM_s016
                                 + 0.8826 w_k1_INTC_s016
                                 + 0.955778 w_k1_KO_s016
                                 + 1.01729 w_k1_MAT_s016
                                 + 0.894389 w_k1_MSFT_s016
                                 + 1.01558 w_k1_PG_s016 >= 1
 z_k1_s0163_constraint:          var_1 + z_k1_s0163 + 1.18124 w_k1_ADM_s016
                                 + 1.01845 w_k1_AXP_s016
                                 + 1.23847 w_k1_DIS_s016
                                 + 1.08484 w_k1_FDX_s016
                                 + 0.783112 w_k1_IBM_s016
                                 + 0.914359 w_k1_INTC_s016
                                 + 0.923346 w_k1_KO_s016
                                 + 0.849537 w_k1_MAT_s016
                                 + 1.00152 w_k1_MSFT_s016
                                 + 1.01031 w_k1_PG_s016 >= 1
 z_k1_s0164_constraint:          var_1 + z_k1_s0164 + 0.654806 w_k1_ADM_s016
                                 + 0.660987 w_k1_AXP_s016
                                 + 0.906796 w_k1_DIS_s016
                                 + 0.861987 w_k1_FDX_s016
                                 + 0.737088 w_k1_IBM_s016
                                 + 0.863341 w_k1_INTC_s016
                                 + 0.804995 w_k1_KO_s016
                                 + 0.743152 w_k1_MAT_s016
                                 + 0.925904 w_k1_MSFT_s016
                                 + 0.903016 w_k1_PG_s016 >= 1
 z_k1_s0165_constraint:          var_1 + z_k1_s0165 + 0.886873 w_k1_ADM_s016
                                 + 0.807213 w_k1_AXP_s016
                                 + 0.661171 w_k1_DIS_s016
                                 + 0.739644 w_k1_FDX_s016
                                 + 0.633756 w_k1_IBM_s016
                                 + 0.815659 w_k1_INTC_s016
                                 + 0.71032 w_k1_KO_s016
                                 + 0.689041 w_k1_MAT_s016
                                 + 0.71726 w_k1_MSFT_s016
                                 + 0.834092 w_k1_PG_s016 >= 1
 z_k1_s0166_constraint:          var_1 + z_k1_s0166 + 0.891264 w_k1_ADM_s016
                                 + 1.07778 w_k1_AXP_s016
                                 + 1.32253 w_k1_DIS_s016
                                 + 0.985828 w_k1_FDX_s016
                                 + 1.30551 w_k1_IBM_s016
                                 + 0.893848 w_k1_INTC_s016
                                 + 1.11654 w_k1_KO_s016 + 1.5622 w_k1_MAT_s016
                                 + 0.954355 w_k1_MSFT_s016
                                 + 1.04196 w_k1_PG_s016 >= 1
 z_k1_s0167_constraint:          var_1 + z_k1_s0167 + 1.23412 w_k1_ADM_s016
                                 + 1.52923 w_k1_AXP_s016
                                 + 1.44926 w_k1_DIS_s016 + 1.3044 w_k1_FDX_s016
                                 + 1.41467 w_k1_IBM_s016
                                 + 1.33649 w_k1_INTC_s016
                                 + 1.36032 w_k1_KO_s016 + 1.87931 w_k1_MAT_s016
                                 + 1.40883 w_k1_MSFT_s016
                                 + 1.26309 w_k1_PG_s016 >= 1
 z_k1_s0170_constraint:          var_1 + z_k1_s0170 + 0.816596 w_k1_ADM_s017
                                 + 0.770769 w_k1_AXP_s017
                                 + 0.647331 w_k1_DIS_s017
                                 + 0.808224 w_k1_FDX_s017
                                 + 0.893675 w_k1_IBM_s017
                                 + 0.77561 w_k1_INTC_s017
                                 + 0.745043 w_k1_KO_s017
                                 + 0.883932 w_k1_MAT_s017
                                 + 0.777927 w_k1_MSFT_s017
                                 + 0.873279 w_k1_PG_s017 >= 1
 z_k1_s0171_constraint:          var_1 + z_k1_s0171 + 1.45289 w_k1_ADM_s017
                                 + 1.66326 w_k1_AXP_s017
                                 + 1.74183 w_k1_DIS_s017
                                 + 1.44524 w_k1_FDX_s017
                                 + 1.14397 w_k1_IBM_s017
                                 + 1.09085 w_k1_INTC_s017 + 1.2637 w_k1_KO_s017
                                 + 1.68896 w_k1_MAT_s017
                                 + 1.2192 w_k1_MSFT_s017 + 1.27296 w_k1_PG_s017
                                 >= 1
 z_k1_s0172_constraint:          var_1 + z_k1_s0172 + 0.863279 w_k1_ADM_s017
                                 + 1.18979 w_k1_AXP_s017
                                 + 0.925027 w_k1_DIS_s017
                                 + 0.973598 w_k1_FDX_s017
                                 + 1.02571 w_k1_IBM_s017
                                 + 0.817302 w_k1_INTC_s017
                                 + 0.973874 w_k1_KO_s017
                                 + 1.85774 w_k1_MAT_s017
                                 + 0.873511 w_k1_MSFT_s017
                                 + 0.979066 w_k1_PG_s017 >= 1
 z_k1_s0173_constraint:          var_1 + z_k1_s0173 + 0.942853 w_k1_ADM_s017
                                 + 0.955861 w_k1_AXP_s017
                                 + 1.39678 w_k1_DIS_s017
                                 + 0.960945 w_k1_FDX_s017
                                 + 1.3221 w_k1_IBM_s017
                                 + 0.979972 w_k1_INTC_s017
                                 + 1.35542 w_k1_KO_s017 + 1.34177 w_k1_MAT_s017
                                 + 1.0677 w_k1_MSFT_s017 + 1.08364 w_k1_PG_s017
                                 >= 1
 z_k1_s0174_constraint:          var_1 + z_k1_s0174 + 1.05062 w_k1_ADM_s017
                                 + 0.999285 w_k1_AXP_s017
                                 + 0.968581 w_k1_DIS_s017
                                 + 1.03974 w_k1_FDX_s017
                                 + 0.716111 w_k1_IBM_s017
                                 + 0.960295 w_k1_INTC_s017
                                 + 0.743772 w_k1_KO_s017
                                 + 0.697574 w_k1_MAT_s017
                                 + 1.13346 w_k1_MSFT_s017
                                 + 0.99599 w_k1_PG_s017 >= 1
 z_k1_s0175_constraint:          var_1 + z_k1_s0175 + 1.09393 w_k1_ADM_s017
                                 + 1.54764 w_k1_AXP_s017
                                 + 1.08811 w_k1_DIS_s017
                                 + 1.12698 w_k1_FDX_s017
                                 + 1.25392 w_k1_IBM_s017
                                 + 1.11199 w_k1_INTC_s017
                                 + 1.06765 w_k1_KO_s017 + 1.89169 w_k1_MAT_s017
                                 + 0.965051 w_k1_MSFT_s017
                                 + 1.15318 w_k1_PG_s017 >= 1
 z_k1_s0176_constraint:          var_1 + z_k1_s0176 + 1.1061 w_k1_ADM_s017
                                 + 1.2543 w_k1_AXP_s017 + 1.51686 w_k1_DIS_s017
                                 + 1.19203 w_k1_FDX_s017
                                 + 1.27296 w_k1_IBM_s017
                                 + 0.985309 w_k1_INTC_s017
                                 + 1.27478 w_k1_KO_s017 + 1.46545 w_k1_MAT_s017
                                 + 1.22443 w_k1_MSFT_s017
                                 + 1.13801 w_k1_PG_s017 >= 1
 z_k1_s0177_constraint:          var_1 + z_k1_s0177 + 0.854401 w_k1_ADM_s017
                                 + 1.02633 w_k1_AXP_s017
                                 + 1.13697 w_k1_DIS_s017
                                 + 1.17638 w_k1_FDX_s017
                                 + 0.873159 w_k1_IBM_s017
                                 + 1.08003 w_k1_INTC_s017
                                 + 0.960378 w_k1_KO_s017
                                 + 1.18873 w_k1_MAT_s017
                                 + 1.20957 w_k1_MSFT_s017
                                 + 0.964389 w_k1_PG_s017 >= 1
 z_k1_s1000_constraint:          var_1 + z_k1_s1000 + 0.836054 w_k1_ADM_s100
                                 + 0.858356 w_k1_AXP_s100
                                 + 0.918649 w_k1_DIS_s100
                                 + 0.896048 w_k1_FDX_s100
                                 + 0.995879 w_k1_IBM_s100
                                 + 1.11842 w_k1_INTC_s100
                                 + 1.13608 w_k1_KO_s100
                                 + 0.990452 w_k1_MAT_s100
                                 + 0.981746 w_k1_MSFT_s100
                                 + 0.912141 w_k1_PG_s100 >= 1
 z_k1_s1001_constraint:          var_1 + z_k1_s1001 + 1.06592 w_k1_ADM_s100
                                 + 1.16552 w_k1_AXP_s100
                                 + 0.957681 w_k1_DIS_s100
                                 + 1.03391 w_k1_FDX_s100
                                 + 1.10201 w_k1_IBM_s100
                                 + 1.03605 w_k1_INTC_s100
                                 + 1.14863 w_k1_KO_s100 + 1.61706 w_k1_MAT_s100
                                 + 0.964079 w_k1_MSFT_s100
                                 + 1.04108 w_k1_PG_s100 >= 1
 z_k1_s1002_constraint:          var_1 + z_k1_s1002 + 0.977749 w_k1_ADM_s100
                                 + 0.764979 w_k1_AXP_s100
                                 + 0.842154 w_k1_DIS_s100
                                 + 0.992748 w_k1_FDX_s100
                                 + 0.666861 w_k1_IBM_s100
                                 + 0.97429 w_k1_INTC_s100
                                 + 0.868514 w_k1_KO_s100
                                 + 0.640919 w_k1_MAT_s100
                                 + 1.00847 w_k1_MSFT_s100
                                 + 0.812264 w_k1_PG_s100 >= 1
 z_k1_s1003_constraint:          var_1 + z_k1_s1003 + 0.930931 w_k1_ADM_s100
                                 + 0.778868 w_k1_AXP_s100
                                 + 0.829322 w_k1_DIS_s100
                                 + 0.901308 w_k1_FDX_s100
                                 + 0.980884 w_k1_IBM_s100
                                 + 0.862427 w_k1_INTC_s100
                                 + 0.959812 w_k1_KO_s100
                                 + 0.816332 w_k1_MAT_s100
                                 + 0.913463 w_k1_MSFT_s100
                                 + 0.897739 w_k1_PG_s100 >= 1
 z_k1_s1004_constraint:          var_1 + z_k1_s1004 + 0.89488 w_k1_ADM_s100
                                 + 0.844713 w_k1_AXP_s100
                                 + 0.892145 w_k1_DIS_s100
                                 + 0.806479 w_k1_FDX_s100
                                 + 1.27538 w_k1_IBM_s100
                                 + 0.994743 w_k1_INTC_s100
                                 + 1.20826 w_k1_KO_s100 + 1.50474 w_k1_MAT_s100
                                 + 0.90508 w_k1_MSFT_s100
                                 + 1.02154 w_k1_PG_s100 >= 1
 z_k1_s1005_constraint:          var_1 + z_k1_s1005 + 0.778333 w_k1_ADM_s100
                                 + 0.618157 w_k1_AXP_s100
                                 + 0.552123 w_k1_DIS_s100
                                 + 0.695212 w_k1_FDX_s100
                                 + 0.735937 w_k1_IBM_s100
                                 + 0.694378 w_k1_INTC_s100
                                 + 0.703592 w_k1_KO_s100
                                 + 0.722501 w_k1_MAT_s100
                                 + 0.788012 w_k1_MSFT_s100
                                 + 0.69325 w_k1_PG_s100 >= 1
 z_k1_s1006_constraint:          var_1 + z_k1_s1006 + 1.07856 w_k1_ADM_s100
                                 + 1.32275 w_k1_AXP_s100
                                 + 1.34317 w_k1_DIS_s100
                                 + 1.32308 w_k1_FDX_s100
                                 + 1.13098 w_k1_IBM_s100
                                 + 1.05162 w_k1_INTC_s100
                                 + 1.27821 w_k1_KO_s100 + 1.66675 w_k1_MAT_s100
                                 + 1.25661 w_k1_MSFT_s100
                                 + 1.14707 w_k1_PG_s100 >= 1
 z_k1_s1007_constraint:          var_1 + z_k1_s1007 + 0.839235 w_k1_ADM_s100
                                 + 1.31473 w_k1_AXP_s100
                                 + 1.54687 w_k1_DIS_s100
                                 + 1.16664 w_k1_FDX_s100
                                 + 1.60034 w_k1_IBM_s100
                                 + 1.1635 w_k1_INTC_s100 + 1.72863 w_k1_KO_s100
                                 + 2.41141 w_k1_MAT_s100
                                 + 1.31787 w_k1_MSFT_s100
                                 + 1.16332 w_k1_PG_s100 >= 1
 z_k1_s1010_constraint:          var_1 + z_k1_s1010 + 0.832381 w_k1_ADM_s101
                                 + 0.718817 w_k1_AXP_s101
                                 + 0.684416 w_k1_DIS_s101
                                 + 0.756854 w_k1_FDX_s101
                                 + 0.737683 w_k1_IBM_s101
                                 + 0.735946 w_k1_INTC_s101
                                 + 0.74411 w_k1_KO_s101
                                 + 0.632836 w_k1_MAT_s101
                                 + 0.756846 w_k1_MSFT_s101
                                 + 0.805574 w_k1_PG_s101 >= 1
 z_k1_s1011_constraint:          var_1 + z_k1_s1011 + 0.820765 w_k1_ADM_s101
                                 + 1.1152 w_k1_AXP_s101 + 1.26907 w_k1_DIS_s101
                                 + 1.0104 w_k1_FDX_s101 + 1.32226 w_k1_IBM_s101
                                 + 1.09074 w_k1_INTC_s101
                                 + 1.24355 w_k1_KO_s101 + 1.10464 w_k1_MAT_s101
                                 + 1.09481 w_k1_MSFT_s101
                                 + 0.957841 w_k1_PG_s101 >= 1
 z_k1_s1012_constraint:          var_1 + z_k1_s1012 + 1.05821 w_k1_ADM_s101
                                 + 1.11165 w_k1_AXP_s101
                                 + 1.07242 w_k1_DIS_s101
                                 + 1.19065 w_k1_FDX_s101
                                 + 0.73432 w_k1_IBM_s101
                                 + 1.00061 w_k1_INTC_s101
                                 + 0.966245 w_k1_KO_s101
                                 + 0.677772 w_k1_MAT_s101
                                 + 1.03861 w_k1_MSFT_s101
                                 + 1.02157 w_k1_PG_s101 >= 1
 z_k1_s1013_constraint:          var_1 + z_k1_s1013 + 0.817935 w_k1_ADM_s101
                                 + 1.31293 w_k1_AXP_s101
                                 + 0.856651 w_k1_DIS_s101
                                 + 1.01146 w_k1_FDX_s101
                                 + 0.86109 w_k1_IBM_s101
                                 + 1.21772 w_k1_INTC_s101
                                 + 0.893209 w_k1_KO_s101
                                 + 0.867965 w_k1_MAT_s101
                                 + 0.892801 w_k1_MSFT_s101
                                 + 0.974351 w_k1_PG_s101 >= 1
 z_k1_s1014_constraint:          var_1 + z_k1_s1014 + 1.05328 w_k1_ADM_s101
                                 + 1.71786 w_k1_AXP_s101
                                 + 1.44803 w_k1_DIS_s101
                                 + 1.32501 w_k1_FDX_s101
                                 + 1.37086 w_k1_IBM_s101
                                 + 1.2961 w_k1_INTC_s101 + 1.39213 w_k1_KO_s101
                                 + 1.71616 w_k1_MAT_s101
                                 + 1.31058 w_k1_MSFT_s101 + 1.0357 w_k1_PG_s101
                                 >= 1
 z_k1_s1015_constraint:          var_1 + z_k1_s1015 + 0.920458 w_k1_ADM_s101
                                 + 1.17677 w_k1_AXP_s101
                                 + 0.801346 w_k1_DIS_s101
                                 + 0.90183 w_k1_FDX_s101 + 1.3399 w_k1_IBM_s101
                                 + 0.889911 w_k1_INTC_s101
                                 + 1.0993 w_k1_KO_s101 + 1.32461 w_k1_MAT_s101
                                 + 0.908864 w_k1_MSFT_s101
                                 + 0.951972 w_k1_PG_s101 >= 1
 z_k1_s1016_constraint:          var_1 + z_k1_s1016 + 1.17597 w_k1_ADM_s101
                                 + 1.10776 w_k1_AXP_s101
                                 + 1.20338 w_k1_DIS_s101
                                 + 1.14584 w_k1_FDX_s101
                                 + 1.04138 w_k1_IBM_s101
                                 + 1.07933 w_k1_INTC_s101 + 1.0732 w_k1_KO_s101
                                 + 0.707657 w_k1_MAT_s101
                                 + 1.08054 w_k1_MSFT_s101
                                 + 0.928018 w_k1_PG_s101 >= 1
 z_k1_s1017_constraint:          var_1 + z_k1_s1017 + 0.657981 w_k1_ADM_s101
                                 + 0.829322 w_k1_AXP_s101
                                 + 0.891765 w_k1_DIS_s101
                                 + 0.826364 w_k1_FDX_s101
                                 + 0.97783 w_k1_IBM_s101
                                 + 0.95865 w_k1_INTC_s101
                                 + 0.979712 w_k1_KO_s101
                                 + 0.866473 w_k1_MAT_s101
                                 + 0.93325 w_k1_MSFT_s101
                                 + 0.898622 w_k1_PG_s101 >= 1
 z_k1_s1020_constraint:          var_1 + z_k1_s1020 + 0.787721 w_k1_ADM_s102
                                 + 0.861483 w_k1_AXP_s102
                                 + 0.81625 w_k1_DIS_s102
                                 + 0.886316 w_k1_FDX_s102
                                 + 0.904882 w_k1_IBM_s102
                                 + 1.04222 w_k1_INTC_s102
                                 + 0.805009 w_k1_KO_s102
                                 + 0.93896 w_k1_MAT_s102
                                 + 0.985122 w_k1_MSFT_s102
                                 + 1.01027 w_k1_PG_s102 >= 1
 z_k1_s1021_constraint:          var_1 + z_k1_s1021 + 0.919385 w_k1_ADM_s102
                                 + 1.39196 w_k1_AXP_s102
                                 + 1.47853 w_k1_DIS_s102
                                 + 1.32493 w_k1_FDX_s102
                                 + 1.16636 w_k1_IBM_s102
                                 + 1.39428 w_k1_INTC_s102
                                 + 1.15167 w_k1_KO_s102 + 1.24584 w_k1_MAT_s102
                                 + 1.32291 w_k1_MSFT_s102
                                 + 1.28781 w_k1_PG_s102 >= 1
 z_k1_s1022_constraint:          var_1 + z_k1_s1022 + 0.750327 w_k1_ADM_s102
                                 + 1.0844 w_k1_AXP_s102 + 1.09741 w_k1_DIS_s102
                                 + 1.00476 w_k1_FDX_s102
                                 + 1.40781 w_k1_IBM_s102
                                 + 1.08135 w_k1_INTC_s102
                                 + 1.21549 w_k1_KO_s102 + 1.38348 w_k1_MAT_s102
                                 + 1.08408 w_k1_MSFT_s102 + 1.188 w_k1_PG_s102
                                 >= 1
 z_k1_s1023_constraint:          var_1 + z_k1_s1023 + 0.624016 w_k1_ADM_s102
                                 + 0.719747 w_k1_AXP_s102
                                 + 0.616987 w_k1_DIS_s102
                                 + 0.768732 w_k1_FDX_s102
                                 + 0.736676 w_k1_IBM_s102
                                 + 0.877253 w_k1_INTC_s102
                                 + 0.7408 w_k1_KO_s102 + 0.68643 w_k1_MAT_s102
                                 + 0.883257 w_k1_MSFT_s102
                                 + 0.803716 w_k1_PG_s102 >= 1
 z_k1_s1024_constraint:          var_1 + z_k1_s1024 + 1.1355 w_k1_ADM_s102
                                 + 1.22404 w_k1_AXP_s102 + 1.276 w_k1_DIS_s102
                                 + 1.24933 w_k1_FDX_s102
                                 + 0.772688 w_k1_IBM_s102
                                 + 1.1562 w_k1_INTC_s102 + 1.00227 w_k1_KO_s102
                                 + 0.605257 w_k1_MAT_s102
                                 + 1.27764 w_k1_MSFT_s102
                                 + 1.05788 w_k1_PG_s102 >= 1
 z_k1_s1025_constraint:          var_1 + z_k1_s1025 + 0.976787 w_k1_ADM_s102
                                 + 0.914089 w_k1_AXP_s102
                                 + 1.34161 w_k1_DIS_s102 + 1.0664 w_k1_FDX_s102
                                 + 1.08535 w_k1_IBM_s102
                                 + 0.969891 w_k1_INTC_s102
                                 + 1.1046 w_k1_KO_s102 + 1.10851 w_k1_MAT_s102
                                 + 1.15475 w_k1_MSFT_s102 + 1.0448 w_k1_PG_s102
                                 >= 1
 z_k1_s1026_constraint:          var_1 + z_k1_s1026 + 0.95493 w_k1_ADM_s102
                                 + 1.24218 w_k1_AXP_s102
                                 + 1.05252 w_k1_DIS_s102
                                 + 1.06739 w_k1_FDX_s102
                                 + 0.988824 w_k1_IBM_s102
                                 + 1.24981 w_k1_INTC_s102
                                 + 0.958393 w_k1_KO_s102
                                 + 1.08455 w_k1_MAT_s102
                                 + 1.0321 w_k1_MSFT_s102 + 1.10462 w_k1_PG_s102
                                 >= 1
 z_k1_s1027_constraint:          var_1 + z_k1_s1027 + 0.736268 w_k1_ADM_s102
                                 + 0.820118 w_k1_AXP_s102
                                 + 0.863172 w_k1_DIS_s102
                                 + 0.951978 w_k1_FDX_s102
                                 + 0.620141 w_k1_IBM_s102
                                 + 1.06993 w_k1_INTC_s102
                                 + 0.757509 w_k1_KO_s102
                                 + 0.716402 w_k1_MAT_s102
                                 + 1.04108 w_k1_MSFT_s102
                                 + 0.923825 w_k1_PG_s102 >= 1
 z_k1_s1030_constraint:          var_1 + z_k1_s1030 + 0.799938 w_k1_ADM_s103
                                 + 1.04106 w_k1_AXP_s103
                                 + 0.845701 w_k1_DIS_s103
                                 + 0.834412 w_k1_FDX_s103
                                 + 1.30788 w_k1_IBM_s103
                                 + 0.978702 w_k1_INTC_s103
                                 + 1.05211 w_k1_KO_s103 + 1.41839 w_k1_MAT_s103
                                 + 0.907549 w_k1_MSFT_s103
                                 + 0.971004 w_k1_PG_s103 >= 1
 z_k1_s1031_constraint:          var_1 + z_k1_s1031 + 1.02438 w_k1_ADM_s103
                                 + 1.15691 w_k1_AXP_s103
                                 + 1.04805 w_k1_DIS_s103 + 1.0364 w_k1_FDX_s103
                                 + 0.710007 w_k1_IBM_s103
                                 + 0.923172 w_k1_INTC_s103
                                 + 0.855873 w_k1_KO_s103
                                 + 0.75637 w_k1_MAT_s103
                                 + 1.05051 w_k1_MSFT_s103
                                 + 0.987492 w_k1_PG_s103 >= 1
 z_k1_s1032_constraint:          var_1 + z_k1_s1032 + 0.868397 w_k1_ADM_s103
                                 + 0.713863 w_k1_AXP_s103
                                 + 0.811121 w_k1_DIS_s103
                                 + 0.74691 w_k1_FDX_s103
                                 + 0.923914 w_k1_IBM_s103
                                 + 0.961845 w_k1_INTC_s103
                                 + 0.84862 w_k1_KO_s103
                                 + 0.985464 w_k1_MAT_s103
                                 + 0.918505 w_k1_MSFT_s103
                                 + 0.90378 w_k1_PG_s103 >= 1
 z_k1_s1033_constraint:          var_1 + z_k1_s1033 + 0.972616 w_k1_ADM_s103
                                 + 1.03728 w_k1_AXP_s103
                                 + 1.23612 w_k1_DIS_s103
                                 + 0.998708 w_k1_FDX_s103
                                 + 1.34303 w_k1_IBM_s103
                                 + 1.12904 w_k1_INTC_s103
                                 + 1.22259 w_k1_KO_s103 + 1.72895 w_k1_MAT_s103
                                 + 1.156 w_k1_MSFT_s103 + 1.04949 w_k1_PG_s103
                                 >= 1
 z_k1_s1034_constraint:          var_1 + z_k1_s1034 + 1.00861 w_k1_ADM_s103
                                 + 1.11347 w_k1_AXP_s103
                                 + 1.35558 w_k1_DIS_s103
                                 + 1.11922 w_k1_FDX_s103
                                 + 1.01852 w_k1_IBM_s103
                                 + 1.05822 w_k1_INTC_s103
                                 + 1.08461 w_k1_KO_s103 + 1.36521 w_k1_MAT_s103
                                 + 1.16226 w_k1_MSFT_s103
                                 + 1.18361 w_k1_PG_s103 >= 1
 z_k1_s1035_constraint:          var_1 + z_k1_s1035 + 1.42267 w_k1_ADM_s103
                                 + 1.2692 w_k1_AXP_s103 + 1.95468 w_k1_DIS_s103
                                 + 1.45192 w_k1_FDX_s103
                                 + 0.971107 w_k1_IBM_s103
                                 + 1.36236 w_k1_INTC_s103
                                 + 1.28382 w_k1_KO_s103 + 1.16541 w_k1_MAT_s103
                                 + 1.52499 w_k1_MSFT_s103
                                 + 1.20525 w_k1_PG_s103 >= 1
 z_k1_s1036_constraint:          var_1 + z_k1_s1036 + 0.842238 w_k1_ADM_s103
                                 + 0.697708 w_k1_AXP_s103
                                 + 0.674972 w_k1_DIS_s103
                                 + 0.794885 w_k1_FDX_s103
                                 + 0.628725 w_k1_IBM_s103
                                 + 0.997495 w_k1_INTC_s103
                                 + 0.755176 w_k1_KO_s103
                                 + 0.703785 w_k1_MAT_s103
                                 + 0.866463 w_k1_MSFT_s103
                                 + 0.838445 w_k1_PG_s103 >= 1
 z_k1_s1037_constraint:          var_1 + z_k1_s1037 + 0.938523 w_k1_ADM_s103
                                 + 1.16511 w_k1_AXP_s103
                                 + 1.20341 w_k1_DIS_s103
                                 + 0.993509 w_k1_FDX_s103
                                 + 1.88968 w_k1_IBM_s103
                                 + 1.25937 w_k1_INTC_s103
                                 + 1.39869 w_k1_KO_s103 + 1.69435 w_k1_MAT_s103
                                 + 1.08181 w_k1_MSFT_s103
                                 + 1.01837 w_k1_PG_s103 >= 1
 z_k1_s1040_constraint:          var_1 + z_k1_s1040 + 1.18273 w_k1_ADM_s104
                                 + 1.18189 w_k1_AXP_s104
                                 + 0.971425 w_k1_DIS_s104 + 1.07 w_k1_FDX_s104
                                 + 0.852974 w_k1_IBM_s104
                                 + 1.19303 w_k1_INTC_s104
                                 + 1.02638 w_k1_KO_s104 + 1.08761 w_k1_MAT_s104
                                 + 1.07866 w_k1_MSFT_s104
                                 + 1.07691 w_k1_PG_s104 >= 1
 z_k1_s1041_constraint:          var_1 + z_k1_s1041 + 0.578448 w_k1_ADM_s104
                                 + 0.539318 w_k1_AXP_s104
                                 + 0.582027 w_k1_DIS_s104
                                 + 0.675111 w_k1_FDX_s104
                                 + 0.660934 w_k1_IBM_s104
                                 + 0.761815 w_k1_INTC_s104
                                 + 0.640601 w_k1_KO_s104
                                 + 0.588746 w_k1_MAT_s104
                                 + 0.745539 w_k1_MSFT_s104
                                 + 0.728444 w_k1_PG_s104 >= 1
 z_k1_s1042_constraint:          var_1 + z_k1_s1042 + 0.877698 w_k1_ADM_s104
                                 + 1.27476 w_k1_AXP_s104
                                 + 1.12184 w_k1_DIS_s104
                                 + 0.951803 w_k1_FDX_s104
                                 + 1.60901 w_k1_IBM_s104
                                 + 1.21634 w_k1_INTC_s104
                                 + 1.38505 w_k1_KO_s104 + 1.79774 w_k1_MAT_s104
                                 + 1.04556 w_k1_MSFT_s104
                                 + 1.04835 w_k1_PG_s104 >= 1
 z_k1_s1043_constraint:          var_1 + z_k1_s1043 + 0.985027 w_k1_ADM_s104
                                 + 0.891386 w_k1_AXP_s104
                                 + 0.846125 w_k1_DIS_s104
                                 + 0.955279 w_k1_FDX_s104
                                 + 0.709854 w_k1_IBM_s104
                                 + 1.09965 w_k1_INTC_s104
                                 + 0.893973 w_k1_KO_s104
                                 + 0.604801 w_k1_MAT_s104
                                 + 1.07913 w_k1_MSFT_s104
                                 + 0.86596 w_k1_PG_s104 >= 1
 z_k1_s1044_constraint:          var_1 + z_k1_s1044 + 0.830156 w_k1_ADM_s104
                                 + 0.858291 w_k1_AXP_s104
                                 + 0.821718 w_k1_DIS_s104
                                 + 0.736478 w_k1_FDX_s104
                                 + 1.08281 w_k1_IBM_s104
                                 + 0.886759 w_k1_INTC_s104
                                 + 1.04415 w_k1_KO_s104 + 1.00478 w_k1_MAT_s104
                                 + 0.828368 w_k1_MSFT_s104
                                 + 0.857303 w_k1_PG_s104 >= 1
 z_k1_s1045_constraint:          var_1 + z_k1_s1045 + 0.928739 w_k1_ADM_s104
                                 + 1.08104 w_k1_AXP_s104
                                 + 0.829966 w_k1_DIS_s104
                                 + 0.884142 w_k1_FDX_s104
                                 + 1.02359 w_k1_IBM_s104
                                 + 1.19826 w_k1_INTC_s104
                                 + 0.97525 w_k1_KO_s104
                                 + 0.946045 w_k1_MAT_s104
                                 + 1.04737 w_k1_MSFT_s104 + 0.8797 w_k1_PG_s104
                                 >= 1
 z_k1_s1046_constraint:          var_1 + z_k1_s1046 + 0.602438 w_k1_ADM_s104
                                 + 0.501972 w_k1_AXP_s104
                                 + 0.768368 w_k1_DIS_s104
                                 + 0.691383 w_k1_FDX_s104
                                 + 0.844404 w_k1_IBM_s104
                                 + 0.834533 w_k1_INTC_s104
                                 + 0.860246 w_k1_KO_s104
                                 + 0.599185 w_k1_MAT_s104
                                 + 0.912426 w_k1_MSFT_s104
                                 + 0.802536 w_k1_PG_s104 >= 1
 z_k1_s1047_constraint:          var_1 + z_k1_s1047 + 1.14402 w_k1_ADM_s104
                                 + 1.31606 w_k1_AXP_s104
                                 + 1.46082 w_k1_DIS_s104
                                 + 1.20408 w_k1_FDX_s104
                                 + 1.08352 w_k1_IBM_s104
                                 + 1.4527 w_k1_INTC_s104 + 1.29185 w_k1_KO_s104
                                 + 0.854394 w_k1_MAT_s104
                                 + 1.26233 w_k1_MSFT_s104
                                 + 1.13161 w_k1_PG_s104 >= 1
 z_k1_s1050_constraint:          var_1 + z_k1_s1050 + 0.992059 w_k1_ADM_s105
                                 + 1.40011 w_k1_AXP_s105
                                 + 1.74157 w_k1_DIS_s105 + 1.4706 w_k1_FDX_s105
                                 + 1.2486 w_k1_IBM_s105
                                 + 1.19884 w_k1_INTC_s105 + 1.3917 w_k1_KO_s105
                                 + 1.38546 w_k1_MAT_s105
                                 + 1.33586 w_k1_MSFT_s105
                                 + 1.31931 w_k1_PG_s105 >= 1
 z_k1_s1051_constraint:          var_1 + z_k1_s1051 + 0.798335 w_k1_ADM_s105
                                 + 1.02482 w_k1_AXP_s105 + 1.2118 w_k1_DIS_s105
                                 + 1.00281 w_k1_FDX_s105
                                 + 1.75505 w_k1_IBM_s105
                                 + 1.11599 w_k1_INTC_s105
                                 + 1.32728 w_k1_KO_s105 + 1.38203 w_k1_MAT_s105
                                 + 1.16082 w_k1_MSFT_s105
                                 + 1.22809 w_k1_PG_s105 >= 1
 z_k1_s1052_constraint:          var_1 + z_k1_s1052 + 0.878365 w_k1_ADM_s105
                                 + 1.28954 w_k1_AXP_s105
                                 + 1.06662 w_k1_DIS_s105 + 1.0453 w_k1_FDX_s105
                                 + 1.3714 w_k1_IBM_s105
                                 + 1.18429 w_k1_INTC_s105 + 1.2844 w_k1_KO_s105
                                 + 1.51442 w_k1_MAT_s105
                                 + 0.924593 w_k1_MSFT_s105
                                 + 1.12851 w_k1_PG_s105 >= 1
 z_k1_s1053_constraint:          var_1 + z_k1_s1053 + 1.17828 w_k1_ADM_s105
                                 + 0.837963 w_k1_AXP_s105
                                 + 0.842574 w_k1_DIS_s105
                                 + 1.13627 w_k1_FDX_s105
                                 + 0.608329 w_k1_IBM_s105
                                 + 1.14685 w_k1_INTC_s105
                                 + 0.678169 w_k1_KO_s105
                                 + 0.426292 w_k1_MAT_s105
                                 + 1.04679 w_k1_MSFT_s105
                                 + 0.872689 w_k1_PG_s105 >= 1
 z_k1_s1054_constraint:          var_1 + z_k1_s1054 + 1.09968 w_k1_ADM_s105
                                 + 1.2028 w_k1_AXP_s105 + 1.00198 w_k1_DIS_s105
                                 + 1.24385 w_k1_FDX_s105
                                 + 1.05221 w_k1_IBM_s105 + 1.038 w_k1_INTC_s105
                                 + 1.04977 w_k1_KO_s105 + 1.23339 w_k1_MAT_s105
                                 + 1.15933 w_k1_MSFT_s105
                                 + 1.01361 w_k1_PG_s105 >= 1
 z_k1_s1055_constraint:          var_1 + z_k1_s1055 + 1.06833 w_k1_ADM_s105
                                 + 0.928634 w_k1_AXP_s105
                                 + 1.16495 w_k1_DIS_s105
                                 + 1.05385 w_k1_FDX_s105
                                 + 0.80565 w_k1_IBM_s105
                                 + 1.18815 w_k1_INTC_s105
                                 + 1.07358 w_k1_KO_s105
                                 + 0.704911 w_k1_MAT_s105
                                 + 1.21366 w_k1_MSFT_s105
                                 + 1.00703 w_k1_PG_s105 >= 1
 z_k1_s1056_constraint:          var_1 + z_k1_s1056 + 0.639697 w_k1_ADM_s105
                                 + 0.783475 w_k1_AXP_s105
                                 + 0.784599 w_k1_DIS_s105
                                 + 0.794844 w_k1_FDX_s105
                                 + 1.0352 w_k1_IBM_s105
                                 + 0.819357 w_k1_INTC_s105
                                 + 0.872321 w_k1_KO_s105
                                 + 1.05434 w_k1_MAT_s105
                                 + 0.836487 w_k1_MSFT_s105
                                 + 0.850246 w_k1_PG_s105 >= 1
 z_k1_s1057_constraint:          var_1 + z_k1_s1057 + 0.763993 w_k1_ADM_s105
                                 + 0.593187 w_k1_AXP_s105
                                 + 0.571908 w_k1_DIS_s105
                                 + 0.716328 w_k1_FDX_s105
                                 + 0.714049 w_k1_IBM_s105
                                 + 0.829892 w_k1_INTC_s105
                                 + 0.683787 w_k1_KO_s105
                                 + 0.690715 w_k1_MAT_s105
                                 + 0.856983 w_k1_MSFT_s105
                                 + 0.811502 w_k1_PG_s105 >= 1
 z_k1_s1060_constraint:          var_1 + z_k1_s1060 + 1.0024 w_k1_ADM_s106
                                 + 1.49822 w_k1_AXP_s106
                                 + 1.21104 w_k1_DIS_s106
                                 + 1.10317 w_k1_FDX_s106
                                 + 1.56638 w_k1_IBM_s106
                                 + 1.33138 w_k1_INTC_s106
                                 + 1.50678 w_k1_KO_s106 + 1.37964 w_k1_MAT_s106
                                 + 1.05098 w_k1_MSFT_s106
                                 + 1.10648 w_k1_PG_s106 >= 1
 z_k1_s1061_constraint:          var_1 + z_k1_s1061 + 0.867154 w_k1_ADM_s106
                                 + 1.01368 w_k1_AXP_s106
                                 + 0.93086 w_k1_DIS_s106
                                 + 0.959414 w_k1_FDX_s106
                                 + 1.08878 w_k1_IBM_s106
                                 + 0.953341 w_k1_INTC_s106
                                 + 1.12355 w_k1_KO_s106 + 1.18868 w_k1_MAT_s106
                                 + 0.974112 w_k1_MSFT_s106
                                 + 1.01497 w_k1_PG_s106 >= 1
 z_k1_s1062_constraint:          var_1 + z_k1_s1062 + 1.01346 w_k1_ADM_s106
                                 + 1.85195 w_k1_AXP_s106
                                 + 1.78424 w_k1_DIS_s106 + 1.4762 w_k1_FDX_s106
                                 + 1.32018 w_k1_IBM_s106
                                 + 1.50058 w_k1_INTC_s106
                                 + 1.65023 w_k1_KO_s106 + 1.46219 w_k1_MAT_s106
                                 + 1.37773 w_k1_MSFT_s106
                                 + 1.21391 w_k1_PG_s106 >= 1
 z_k1_s1063_constraint:          var_1 + z_k1_s1063 + 0.987841 w_k1_ADM_s106
                                 + 1.4303 w_k1_AXP_s106 + 1.23428 w_k1_DIS_s106
                                 + 1.24828 w_k1_FDX_s106
                                 + 0.812486 w_k1_IBM_s106
                                 + 1.41097 w_k1_INTC_s106
                                 + 1.03102 w_k1_KO_s106 + 1.07882 w_k1_MAT_s106
                                 + 1.21586 w_k1_MSFT_s106
                                 + 1.05941 w_k1_PG_s106 >= 1
 z_k1_s1064_constraint:          var_1 + z_k1_s1064 + 0.740896 w_k1_ADM_s106
                                 + 0.922682 w_k1_AXP_s106
                                 + 0.763079 w_k1_DIS_s106
                                 + 0.752023 w_k1_FDX_s106
                                 + 1.1051 w_k1_IBM_s106
                                 + 1.06028 w_k1_INTC_s106
                                 + 1.11147 w_k1_KO_s106
                                 + 0.934416 w_k1_MAT_s106
                                 + 0.886705 w_k1_MSFT_s106
                                 + 1.01742 w_k1_PG_s106 >= 1
 z_k1_s1065_constraint:          var_1 + z_k1_s1065 + 0.877519 w_k1_ADM_s106
                                 + 1.36573 w_k1_AXP_s106
                                 + 1.38175 w_k1_DIS_s106
                                 + 1.12232 w_k1_FDX_s106
                                 + 1.25908 w_k1_IBM_s106
                                 + 1.21819 w_k1_INTC_s106
                                 + 1.31614 w_k1_KO_s106 + 1.38244 w_k1_MAT_s106
                                 + 1.17938 w_k1_MSFT_s106
                                 + 1.10851 w_k1_PG_s106 >= 1
 z_k1_s1066_constraint:          var_1 + z_k1_s1066 + 0.816437 w_k1_ADM_s106
                                 + 0.794653 w_k1_AXP_s106
                                 + 0.808959 w_k1_DIS_s106
                                 + 0.879167 w_k1_FDX_s106
                                 + 0.740367 w_k1_IBM_s106
                                 + 0.866493 w_k1_INTC_s106
                                 + 0.835358 w_k1_KO_s106
                                 + 0.60761 w_k1_MAT_s106
                                 + 0.881708 w_k1_MSFT_s106
                                 + 0.830331 w_k1_PG_s106 >= 1
 z_k1_s1067_constraint:          var_1 + z_k1_s1067 + 1.04663 w_k1_ADM_s106
                                 + 1.06861 w_k1_AXP_s106
                                 + 1.24514 w_k1_DIS_s106
                                 + 1.16544 w_k1_FDX_s106
                                 + 1.02931 w_k1_IBM_s106
                                 + 1.20754 w_k1_INTC_s106
                                 + 1.24015 w_k1_KO_s106
                                 + 0.768101 w_k1_MAT_s106
                                 + 1.21592 w_k1_MSFT_s106
                                 + 1.09607 w_k1_PG_s106 >= 1
 z_k1_s1070_constraint:          var_1 + z_k1_s1070 + 1.47147 w_k1_ADM_s107
                                 + 1.22162 w_k1_AXP_s107
                                 + 1.37803 w_k1_DIS_s107
                                 + 1.23313 w_k1_FDX_s107
                                 + 1.14026 w_k1_IBM_s107
                                 + 1.16914 w_k1_INTC_s107
                                 + 1.39082 w_k1_KO_s107 + 1.18083 w_k1_MAT_s107
                                 + 1.28552 w_k1_MSFT_s107
                                 + 1.13638 w_k1_PG_s107 >= 1
 z_k1_s1071_constraint:          var_1 + z_k1_s1071 + 1.13836 w_k1_ADM_s107
                                 + 0.937302 w_k1_AXP_s107
                                 + 0.980366 w_k1_DIS_s107
                                 + 1.07076 w_k1_FDX_s107
                                 + 0.852761 w_k1_IBM_s107
                                 + 0.951674 w_k1_INTC_s107
                                 + 0.957332 w_k1_KO_s107
                                 + 1.00112 w_k1_MAT_s107
                                 + 1.01175 w_k1_MSFT_s107
                                 + 0.933405 w_k1_PG_s107 >= 1
 z_k1_s1072_constraint:          var_1 + z_k1_s1072 + 0.876968 w_k1_ADM_s107
                                 + 0.685511 w_k1_AXP_s107
                                 + 0.799871 w_k1_DIS_s107
                                 + 0.905062 w_k1_FDX_s107
                                 + 0.552621 w_k1_IBM_s107
                                 + 0.902488 w_k1_INTC_s107
                                 + 0.789522 w_k1_KO_s107
                                 + 0.427636 w_k1_MAT_s107
                                 + 1.02801 w_k1_MSFT_s107
                                 + 0.809718 w_k1_PG_s107 >= 1
 z_k1_s1073_constraint:          var_1 + z_k1_s1073 + 1.07755 w_k1_ADM_s107
                                 + 1.12703 w_k1_AXP_s107
                                 + 1.12985 w_k1_DIS_s107
                                 + 1.05457 w_k1_FDX_s107 + 1.5017 w_k1_IBM_s107
                                 + 1.20675 w_k1_INTC_s107
                                 + 1.28531 w_k1_KO_s107 + 1.61562 w_k1_MAT_s107
                                 + 1.04641 w_k1_MSFT_s107 + 1.0775 w_k1_PG_s107
                                 >= 1
 z_k1_s1074_constraint:          var_1 + z_k1_s1074 + 1.18949 w_k1_ADM_s107
                                 + 0.869595 w_k1_AXP_s107
                                 + 0.818022 w_k1_DIS_s107
                                 + 0.909354 w_k1_FDX_s107
                                 + 1.02216 w_k1_IBM_s107
                                 + 0.966335 w_k1_INTC_s107
                                 + 1.09459 w_k1_KO_s107
                                 + 0.814696 w_k1_MAT_s107
                                 + 0.870008 w_k1_MSFT_s107
                                 + 0.877592 w_k1_PG_s107 >= 1
 z_k1_s1075_constraint:          var_1 + z_k1_s1075 + 1.32156 w_k1_ADM_s107
                                 + 1.06119 w_k1_AXP_s107
                                 + 1.02988 w_k1_DIS_s107
                                 + 1.15521 w_k1_FDX_s107
                                 + 0.621539 w_k1_IBM_s107
                                 + 1.04165 w_k1_INTC_s107
                                 + 0.936682 w_k1_KO_s107
                                 + 0.682457 w_k1_MAT_s107
                                 + 1.17253 w_k1_MSFT_s107
                                 + 1.04854 w_k1_PG_s107 >= 1
 z_k1_s1076_constraint:          var_1 + z_k1_s1076 + 1.15729 w_k1_ADM_s107
                                 + 1.12495 w_k1_AXP_s107
                                 + 1.17559 w_k1_DIS_s107
                                 + 1.23347 w_k1_FDX_s107
                                 + 0.838716 w_k1_IBM_s107
                                 + 1.41468 w_k1_INTC_s107
                                 + 1.03073 w_k1_KO_s107
                                 + 0.823678 w_k1_MAT_s107
                                 + 1.28066 w_k1_MSFT_s107
                                 + 0.98557 w_k1_PG_s107 >= 1
 z_k1_s1077_constraint:          var_1 + z_k1_s1077 + 0.920917 w_k1_ADM_s107
                                 + 0.781007 w_k1_AXP_s107
                                 + 0.762853 w_k1_DIS_s107
                                 + 0.855905 w_k1_FDX_s107
                                 + 0.774668 w_k1_IBM_s107
                                 + 0.892881 w_k1_INTC_s107
                                 + 0.902458 w_k1_KO_s107
                                 + 0.915414 w_k1_MAT_s107
                                 + 0.940766 w_k1_MSFT_s107
                                 + 0.909949 w_k1_PG_s107 >= 1
 z_k1_s1100_constraint:          var_1 + z_k1_s1100 + 0.937143 w_k1_ADM_s110
                                 + 0.997848 w_k1_AXP_s110
                                 + 0.887863 w_k1_DIS_s110
                                 + 0.946845 w_k1_FDX_s110
                                 + 0.825207 w_k1_IBM_s110
                                 + 0.849377 w_k1_INTC_s110
                                 + 0.879622 w_k1_KO_s110
                                 + 1.07179 w_k1_MAT_s110
                                 + 0.985703 w_k1_MSFT_s110
                                 + 1.06356 w_k1_PG_s110 >= 1
 z_k1_s1101_constraint:          var_1 + z_k1_s1101 + 1.14274 w_k1_ADM_s110
                                 + 1.62068 w_k1_AXP_s110 + 1.4831 w_k1_DIS_s110
                                 + 1.2814 w_k1_FDX_s110 + 1.37235 w_k1_IBM_s110
                                 + 1.31929 w_k1_INTC_s110
                                 + 1.28979 w_k1_KO_s110 + 1.32967 w_k1_MAT_s110
                                 + 1.13387 w_k1_MSFT_s110
                                 + 1.32831 w_k1_PG_s110 >= 1
 z_k1_s1102_constraint:          var_1 + z_k1_s1102 + 0.930749 w_k1_ADM_s110
                                 + 0.785865 w_k1_AXP_s110
                                 + 0.548522 w_k1_DIS_s110
                                 + 0.702589 w_k1_FDX_s110
                                 + 0.863305 w_k1_IBM_s110
                                 + 0.703381 w_k1_INTC_s110
                                 + 0.696189 w_k1_KO_s110
                                 + 0.811871 w_k1_MAT_s110
                                 + 0.732239 w_k1_MSFT_s110
                                 + 0.914261 w_k1_PG_s110 >= 1
 z_k1_s1103_constraint:          var_1 + z_k1_s1103 + 1.14583 w_k1_ADM_s110
                                 + 1.09444 w_k1_AXP_s110
                                 + 1.24045 w_k1_DIS_s110
                                 + 1.09616 w_k1_FDX_s110
                                 + 1.08973 w_k1_IBM_s110
                                 + 1.02801 w_k1_INTC_s110
                                 + 1.28327 w_k1_KO_s110 + 1.14832 w_k1_MAT_s110
                                 + 1.13816 w_k1_MSFT_s110
                                 + 1.22041 w_k1_PG_s110 >= 1
 z_k1_s1104_constraint:          var_1 + z_k1_s1104 + 0.867934 w_k1_ADM_s110
                                 + 0.697122 w_k1_AXP_s110
                                 + 0.654694 w_k1_DIS_s110
                                 + 0.832364 w_k1_FDX_s110
                                 + 0.704083 w_k1_IBM_s110
                                 + 0.979001 w_k1_INTC_s110
                                 + 0.746868 w_k1_KO_s110
                                 + 0.533133 w_k1_MAT_s110
                                 + 0.887395 w_k1_MSFT_s110
                                 + 0.920498 w_k1_PG_s110 >= 1
 z_k1_s1105_constraint:          var_1 + z_k1_s1105 + 1.03148 w_k1_ADM_s110
                                 + 1.13702 w_k1_AXP_s110
                                 + 0.724476 w_k1_DIS_s110
                                 + 0.926479 w_k1_FDX_s110
                                 + 0.901522 w_k1_IBM_s110
                                 + 0.953661 w_k1_INTC_s110
                                 + 0.943559 w_k1_KO_s110
                                 + 1.12007 w_k1_MAT_s110
                                 + 0.892695 w_k1_MSFT_s110
                                 + 1.02995 w_k1_PG_s110 >= 1
 z_k1_s1106_constraint:          var_1 + z_k1_s1106 + 1.07777 w_k1_ADM_s110
                                 + 0.934059 w_k1_AXP_s110
                                 + 1.15281 w_k1_DIS_s110
                                 + 1.08412 w_k1_FDX_s110
                                 + 0.748359 w_k1_IBM_s110
                                 + 1.05721 w_k1_INTC_s110
                                 + 0.950239 w_k1_KO_s110
                                 + 0.589654 w_k1_MAT_s110
                                 + 1.07453 w_k1_MSFT_s110
                                 + 1.07509 w_k1_PG_s110 >= 1
 z_k1_s1107_constraint:          var_1 + z_k1_s1107 + 0.889679 w_k1_ADM_s110
                                 + 0.952172 w_k1_AXP_s110
                                 + 0.882469 w_k1_DIS_s110
                                 + 0.855331 w_k1_FDX_s110
                                 + 1.16138 w_k1_IBM_s110
                                 + 0.840745 w_k1_INTC_s110
                                 + 1.08797 w_k1_KO_s110 + 1.02921 w_k1_MAT_s110
                                 + 0.872458 w_k1_MSFT_s110
                                 + 0.926599 w_k1_PG_s110 >= 1
 z_k1_s1110_constraint:          var_1 + z_k1_s1110 + 1.32584 w_k1_ADM_s111
                                 + 1.02682 w_k1_AXP_s111
                                 + 0.966581 w_k1_DIS_s111
                                 + 0.975504 w_k1_FDX_s111
                                 + 0.973446 w_k1_IBM_s111
                                 + 1.10209 w_k1_INTC_s111
                                 + 0.956755 w_k1_KO_s111
                                 + 0.876659 w_k1_MAT_s111
                                 + 0.958362 w_k1_MSFT_s111
                                 + 0.970773 w_k1_PG_s111 >= 1
 z_k1_s1111_constraint:          var_1 + z_k1_s1111 + 1.24012 w_k1_ADM_s111
                                 + 1.4216 w_k1_AXP_s111
                                 + 0.970685 w_k1_DIS_s111
                                 + 1.09887 w_k1_FDX_s111
                                 + 0.701545 w_k1_IBM_s111
                                 + 0.912738 w_k1_INTC_s111
                                 + 0.864593 w_k1_KO_s111
                                 + 1.53466 w_k1_MAT_s111
                                 + 0.960789 w_k1_MSFT_s111
                                 + 0.928985 w_k1_PG_s111 >= 1
 z_k1_s1112_constraint:          var_1 + z_k1_s1112 + 1.19851 w_k1_ADM_s111
                                 + 1.33293 w_k1_AXP_s111 + 1.4277 w_k1_DIS_s111
                                 + 1.1246 w_k1_FDX_s111 + 1.28598 w_k1_IBM_s111
                                 + 1.18757 w_k1_INTC_s111
                                 + 1.24972 w_k1_KO_s111 + 1.32313 w_k1_MAT_s111
                                 + 1.12891 w_k1_MSFT_s111
                                 + 1.01242 w_k1_PG_s111 >= 1
 z_k1_s1113_constraint:          var_1 + z_k1_s1113 + 1.45794 w_k1_ADM_s111
                                 + 1.06302 w_k1_AXP_s111
                                 + 1.57952 w_k1_DIS_s111
                                 + 1.28136 w_k1_FDX_s111
                                 + 1.02725 w_k1_IBM_s111
                                 + 1.41601 w_k1_INTC_s111
                                 + 1.23594 w_k1_KO_s111
                                 + 0.912993 w_k1_MAT_s111
                                 + 1.26431 w_k1_MSFT_s111
                                 + 1.07104 w_k1_PG_s111 >= 1
 z_k1_s1114_constraint:          var_1 + z_k1_s1114 + 1.24829 w_k1_ADM_s111
                                 + 1.42844 w_k1_AXP_s111
                                 + 1.26075 w_k1_DIS_s111
                                 + 1.07971 w_k1_FDX_s111
                                 + 1.64551 w_k1_IBM_s111
                                 + 1.16907 w_k1_INTC_s111
                                 + 1.39936 w_k1_KO_s111 + 1.95594 w_k1_MAT_s111
                                 + 1.15293 w_k1_MSFT_s111
                                 + 1.21945 w_k1_PG_s111 >= 1
 z_k1_s1115_constraint:          var_1 + z_k1_s1115 + 0.909261 w_k1_ADM_s111
                                 + 0.816049 w_k1_AXP_s111
                                 + 0.961823 w_k1_DIS_s111
                                 + 0.870664 w_k1_FDX_s111
                                 + 1.23842 w_k1_IBM_s111
                                 + 0.954974 w_k1_INTC_s111
                                 + 1.01033 w_k1_KO_s111 + 1.13957 w_k1_MAT_s111
                                 + 0.832311 w_k1_MSFT_s111
                                 + 0.930374 w_k1_PG_s111 >= 1
 z_k1_s1116_constraint:          var_1 + z_k1_s1116 + 0.890036 w_k1_ADM_s111
                                 + 0.704906 w_k1_AXP_s111
                                 + 0.66881 w_k1_DIS_s111
                                 + 0.788127 w_k1_FDX_s111
                                 + 0.670027 w_k1_IBM_s111
                                 + 0.835245 w_k1_INTC_s111
                                 + 0.744984 w_k1_KO_s111
                                 + 0.619143 w_k1_MAT_s111
                                 + 0.753702 w_k1_MSFT_s111
                                 + 0.799973 w_k1_PG_s111 >= 1
 z_k1_s1117_constraint:          var_1 + z_k1_s1117 + 0.933533 w_k1_ADM_s111
                                 + 0.820238 w_k1_AXP_s111
                                 + 0.913064 w_k1_DIS_s111
                                 + 0.991274 w_k1_FDX_s111
                                 + 0.672846 w_k1_IBM_s111
                                 + 0.998741 w_k1_INTC_s111
                                 + 0.838143 w_k1_KO_s111
                                 + 0.984779 w_k1_MAT_s111
                                 + 1.01925 w_k1_MSFT_s111
                                 + 0.905807 w_k1_PG_s111 >= 1
 z_k1_s1120_constraint:          var_1 + z_k1_s1120 + 0.7506 w_k1_ADM_s112
                                 + 0.955854 w_k1_AXP_s112
                                 + 0.989243 w_k1_DIS_s112
                                 + 0.81954 w_k1_FDX_s112
                                 + 1.28978 w_k1_IBM_s112
                                 + 0.898961 w_k1_INTC_s112
                                 + 1.03434 w_k1_KO_s112 + 1.2917 w_k1_MAT_s112
                                 + 0.858408 w_k1_MSFT_s112
                                 + 0.934185 w_k1_PG_s112 >= 1
 z_k1_s1121_constraint:          var_1 + z_k1_s1121 + 0.851677 w_k1_ADM_s112
                                 + 0.902878 w_k1_AXP_s112
                                 + 0.774625 w_k1_DIS_s112
                                 + 0.879422 w_k1_FDX_s112
                                 + 0.946574 w_k1_IBM_s112
                                 + 0.940448 w_k1_INTC_s112
                                 + 1.00918 w_k1_KO_s112
                                 + 0.723958 w_k1_MAT_s112
                                 + 0.911133 w_k1_MSFT_s112
                                 + 0.878405 w_k1_PG_s112 >= 1
 z_k1_s1122_constraint:          var_1 + z_k1_s1122 + 1.1625 w_k1_ADM_s112
                                 + 1.36345 w_k1_AXP_s112
                                 + 1.61785 w_k1_DIS_s112
                                 + 1.28529 w_k1_FDX_s112
                                 + 1.30419 w_k1_IBM_s112
                                 + 1.36319 w_k1_INTC_s112
                                 + 1.32423 w_k1_KO_s112 + 1.17514 w_k1_MAT_s112
                                 + 1.22488 w_k1_MSFT_s112
                                 + 1.12613 w_k1_PG_s112 >= 1
 z_k1_s1123_constraint:          var_1 + z_k1_s1123 + 1.12684 w_k1_ADM_s112
                                 + 1.13649 w_k1_AXP_s112
                                 + 1.01682 w_k1_DIS_s112
                                 + 1.14895 w_k1_FDX_s112
                                 + 0.694203 w_k1_IBM_s112
                                 + 0.921976 w_k1_INTC_s112
                                 + 0.841209 w_k1_KO_s112
                                 + 0.704622 w_k1_MAT_s112
                                 + 0.91454 w_k1_MSFT_s112
                                 + 0.966144 w_k1_PG_s112 >= 1
 z_k1_s1124_constraint:          var_1 + z_k1_s1124 + 1.11874 w_k1_ADM_s112
                                 + 1.53675 w_k1_AXP_s112
                                 + 1.40937 w_k1_DIS_s112
                                 + 1.16405 w_k1_FDX_s112
                                 + 1.73211 w_k1_IBM_s112
                                 + 1.19316 w_k1_INTC_s112
                                 + 1.42921 w_k1_KO_s112 + 1.7775 w_k1_MAT_s112
                                 + 1.06633 w_k1_MSFT_s112
                                 + 1.13351 w_k1_PG_s112 >= 1
 z_k1_s1125_constraint:          var_1 + z_k1_s1125 + 0.834119 w_k1_ADM_s112
                                 + 0.704446 w_k1_AXP_s112
                                 + 0.770076 w_k1_DIS_s112
                                 + 0.851319 w_k1_FDX_s112
                                 + 0.629135 w_k1_IBM_s112
                                 + 0.923601 w_k1_INTC_s112
                                 + 0.734765 w_k1_KO_s112
                                 + 0.460796 w_k1_MAT_s112
                                 + 0.931189 w_k1_MSFT_s112
                                 + 0.81951 w_k1_PG_s112 >= 1
 z_k1_s1126_constraint:          var_1 + z_k1_s1126 + 0.571777 w_k1_ADM_s112
                                 + 0.54293 w_k1_AXP_s112
                                 + 0.623849 w_k1_DIS_s112
                                 + 0.636361 w_k1_FDX_s112
                                 + 0.691541 w_k1_IBM_s112
                                 + 0.768648 w_k1_INTC_s112
                                 + 0.673449 w_k1_KO_s112
                                 + 0.580743 w_k1_MAT_s112
                                 + 0.734174 w_k1_MSFT_s112
                                 + 0.693566 w_k1_PG_s112 >= 1
 z_k1_s1127_constraint:          var_1 + z_k1_s1127 + 1.22663 w_k1_ADM_s112
                                 + 1.23496 w_k1_AXP_s112
                                 + 1.33341 w_k1_DIS_s112
                                 + 1.21021 w_k1_FDX_s112
                                 + 0.980253 w_k1_IBM_s112
                                 + 1.17116 w_k1_INTC_s112
                                 + 1.09279 w_k1_KO_s112
                                 + 0.955912 w_k1_MAT_s112
                                 + 1.15648 w_k1_MSFT_s112
                                 + 1.09274 w_k1_PG_s112 >= 1
 z_k1_s1130_constraint:          var_1 + z_k1_s1130 + 0.723941 w_k1_ADM_s113
                                 + 0.652926 w_k1_AXP_s113
                                 + 0.643769 w_k1_DIS_s113
                                 + 0.784978 w_k1_FDX_s113
                                 + 0.755141 w_k1_IBM_s113
                                 + 0.736179 w_k1_INTC_s113
                                 + 0.744478 w_k1_KO_s113
                                 + 0.749892 w_k1_MAT_s113
                                 + 0.882498 w_k1_MSFT_s113
                                 + 0.841283 w_k1_PG_s113 >= 1
 z_k1_s1131_constraint:          var_1 + z_k1_s1131 + 1.47639 w_k1_ADM_s113
                                 + 1.37982 w_k1_AXP_s113
                                 + 1.67417 w_k1_DIS_s113
                                 + 1.35867 w_k1_FDX_s113
                                 + 0.888778 w_k1_IBM_s113
                                 + 1.037 w_k1_INTC_s113 + 1.3287 w_k1_KO_s113
                                 + 0.862319 w_k1_MAT_s113
                                 + 1.26645 w_k1_MSFT_s113
                                 + 1.28443 w_k1_PG_s113 >= 1
 z_k1_s1132_constraint:          var_1 + z_k1_s1132 + 1.08151 w_k1_ADM_s113
                                 + 0.988934 w_k1_AXP_s113
                                 + 1.12606 w_k1_DIS_s113
                                 + 1.01031 w_k1_FDX_s113
                                 + 1.08647 w_k1_IBM_s113
                                 + 0.803547 w_k1_INTC_s113
                                 + 1.22795 w_k1_KO_s113 + 1.25015 w_k1_MAT_s113
                                 + 1.12704 w_k1_MSFT_s113
                                 + 1.04326 w_k1_PG_s113 >= 1
 z_k1_s1133_constraint:          var_1 + z_k1_s1133 + 1.49369 w_k1_ADM_s113
                                 + 1.41563 w_k1_AXP_s113
                                 + 1.55752 w_k1_DIS_s113
                                 + 1.21967 w_k1_FDX_s113
                                 + 1.46036 w_k1_IBM_s113
                                 + 1.42254 w_k1_INTC_s113 + 1.4325 w_k1_KO_s113
                                 + 1.24775 w_k1_MAT_s113
                                 + 1.25555 w_k1_MSFT_s113
                                 + 1.19961 w_k1_PG_s113 >= 1
 z_k1_s1134_constraint:          var_1 + z_k1_s1134 + 0.680113 w_k1_ADM_s113
                                 + 0.916056 w_k1_AXP_s113
                                 + 0.823031 w_k1_DIS_s113
                                 + 0.822112 w_k1_FDX_s113
                                 + 1.15014 w_k1_IBM_s113
                                 + 0.730197 w_k1_INTC_s113
                                 + 1.08703 w_k1_KO_s113 + 1.28132 w_k1_MAT_s113
                                 + 0.875558 w_k1_MSFT_s113
                                 + 0.848312 w_k1_PG_s113 >= 1
 z_k1_s1135_constraint:          var_1 + z_k1_s1135 + 0.881424 w_k1_ADM_s113
                                 + 0.604556 w_k1_AXP_s113
                                 + 0.71927 w_k1_DIS_s113
                                 + 0.807023 w_k1_FDX_s113
                                 + 0.739631 w_k1_IBM_s113
                                 + 0.687816 w_k1_INTC_s113
                                 + 0.840461 w_k1_KO_s113
                                 + 0.59778 w_k1_MAT_s113
                                 + 0.936458 w_k1_MSFT_s113
                                 + 0.903708 w_k1_PG_s113 >= 1
 z_k1_s1136_constraint:          var_1 + z_k1_s1136 + 1.07365 w_k1_ADM_s113
                                 + 1.03957 w_k1_AXP_s113
                                 + 0.820529 w_k1_DIS_s113
                                 + 1.02478 w_k1_FDX_s113
                                 + 0.826465 w_k1_IBM_s113
                                 + 1.00983 w_k1_INTC_s113
                                 + 0.928475 w_k1_KO_s113
                                 + 0.821918 w_k1_MAT_s113
                                 + 1.04509 w_k1_MSFT_s113
                                 + 0.948347 w_k1_PG_s113 >= 1
 z_k1_s1137_constraint:          var_1 + z_k1_s1137 + 1.28731 w_k1_ADM_s113
                                 + 1.29515 w_k1_AXP_s113
                                 + 1.03919 w_k1_DIS_s113
                                 + 1.17859 w_k1_FDX_s113
                                 + 1.00437 w_k1_IBM_s113
                                 + 1.09424 w_k1_INTC_s113 + 1.0542 w_k1_KO_s113
                                 + 1.32696 w_k1_MAT_s113
                                 + 1.1665 w_k1_MSFT_s113 + 1.14939 w_k1_PG_s113
                                 >= 1
 z_k1_s1140_constraint:          var_1 + z_k1_s1140 + 1.22024 w_k1_ADM_s114
                                 + 1.7259 w_k1_AXP_s114 + 1.7501 w_k1_DIS_s114
                                 + 1.46243 w_k1_FDX_s114
                                 + 1.18671 w_k1_IBM_s114
                                 + 1.4005 w_k1_INTC_s114 + 1.56363 w_k1_KO_s114
                                 + 1.81978 w_k1_MAT_s114
                                 + 1.42891 w_k1_MSFT_s114
                                 + 1.34298 w_k1_PG_s114 >= 1
 z_k1_s1141_constraint:          var_1 + z_k1_s1141 + 1.22796 w_k1_ADM_s114
                                 + 0.897945 w_k1_AXP_s114
                                 + 1.09652 w_k1_DIS_s114
                                 + 1.12609 w_k1_FDX_s114
                                 + 0.797191 w_k1_IBM_s114
                                 + 0.868419 w_k1_INTC_s114
                                 + 1.03688 w_k1_KO_s114
                                 + 0.740147 w_k1_MAT_s114
                                 + 1.08708 w_k1_MSFT_s114
                                 + 1.02586 w_k1_PG_s114 >= 1
 z_k1_s1142_constraint:          var_1 + z_k1_s1142 + 1.04585 w_k1_ADM_s114
                                 + 1.21807 w_k1_AXP_s114
                                 + 1.20324 w_k1_DIS_s114
                                 + 1.09904 w_k1_FDX_s114 + 1.489 w_k1_IBM_s114
                                 + 1.02648 w_k1_INTC_s114
                                 + 1.28021 w_k1_KO_s114 + 1.83859 w_k1_MAT_s114
                                 + 1.10705 w_k1_MSFT_s114
                                 + 1.17621 w_k1_PG_s114 >= 1
 z_k1_s1143_constraint:          var_1 + z_k1_s1143 + 0.674313 w_k1_ADM_s114
                                 + 0.850883 w_k1_AXP_s114
                                 + 0.715862 w_k1_DIS_s114
                                 + 0.789803 w_k1_FDX_s114
                                 + 1.07925 w_k1_IBM_s114
                                 + 0.746878 w_k1_INTC_s114
                                 + 0.867019 w_k1_KO_s114
                                 + 0.975448 w_k1_MAT_s114
                                 + 0.801556 w_k1_MSFT_s114
                                 + 0.891317 w_k1_PG_s114 >= 1
 z_k1_s1144_constraint:          var_1 + z_k1_s1144 + 0.907101 w_k1_ADM_s114
                                 + 0.69465 w_k1_AXP_s114
                                 + 0.655718 w_k1_DIS_s114
                                 + 0.892299 w_k1_FDX_s114
                                 + 0.617375 w_k1_IBM_s114
                                 + 0.875092 w_k1_INTC_s114
                                 + 0.741628 w_k1_KO_s114
                                 + 0.694035 w_k1_MAT_s114
                                 + 0.853005 w_k1_MSFT_s114
                                 + 0.890287 w_k1_PG_s114 >= 1
 z_k1_s1145_constraint:          var_1 + z_k1_s1145 + 1.11962 w_k1_ADM_s114
                                 + 0.898775 w_k1_AXP_s114
                                 + 1.02148 w_k1_DIS_s114
                                 + 0.909623 w_k1_FDX_s114
                                 + 1.24862 w_k1_IBM_s114
                                 + 0.831991 w_k1_INTC_s114
                                 + 1.26096 w_k1_KO_s114 + 1.41218 w_k1_MAT_s114
                                 + 0.990883 w_k1_MSFT_s114
                                 + 1.11199 w_k1_PG_s114 >= 1
 z_k1_s1146_constraint:          var_1 + z_k1_s1146 + 1.2817 w_k1_ADM_s114
                                 + 1.10049 w_k1_AXP_s114
                                 + 1.22511 w_k1_DIS_s114 + 1.257 w_k1_FDX_s114
                                 + 0.935035 w_k1_IBM_s114
                                 + 1.00784 w_k1_INTC_s114
                                 + 1.19792 w_k1_KO_s114 + 1.07643 w_k1_MAT_s114
                                 + 1.28414 w_k1_MSFT_s114
                                 + 1.19897 w_k1_PG_s114 >= 1
 z_k1_s1147_constraint:          var_1 + z_k1_s1147 + 1.04305 w_k1_ADM_s114
                                 + 0.896387 w_k1_AXP_s114
                                 + 0.962248 w_k1_DIS_s114
                                 + 0.889526 w_k1_FDX_s114
                                 + 0.984719 w_k1_IBM_s114
                                 + 0.8923 w_k1_INTC_s114
                                 + 0.935274 w_k1_KO_s114
                                 + 0.807996 w_k1_MAT_s114
                                 + 0.993025 w_k1_MSFT_s114
                                 + 1.02738 w_k1_PG_s114 >= 1
 z_k1_s1150_constraint:          var_1 + z_k1_s1150 + 0.986336 w_k1_ADM_s115
                                 + 0.83368 w_k1_AXP_s115
                                 + 0.962204 w_k1_DIS_s115
                                 + 0.981168 w_k1_FDX_s115
                                 + 0.939553 w_k1_IBM_s115
                                 + 0.997299 w_k1_INTC_s115
                                 + 0.903058 w_k1_KO_s115
                                 + 0.693817 w_k1_MAT_s115
                                 + 1.01187 w_k1_MSFT_s115
                                 + 1.02467 w_k1_PG_s115 >= 1
 z_k1_s1151_constraint:          var_1 + z_k1_s1151 + 1.28252 w_k1_ADM_s115
                                 + 1.09238 w_k1_AXP_s115
                                 + 1.25679 w_k1_DIS_s115
                                 + 1.10247 w_k1_FDX_s115
                                 + 1.01072 w_k1_IBM_s115
                                 + 1.04854 w_k1_INTC_s115
                                 + 0.982166 w_k1_KO_s115
                                 + 0.717998 w_k1_MAT_s115
                                 + 1.03591 w_k1_MSFT_s115
                                 + 1.05906 w_k1_PG_s115 >= 1
 z_k1_s1152_constraint:          var_1 + z_k1_s1152 + 0.866012 w_k1_ADM_s115
                                 + 0.567317 w_k1_AXP_s115
                                 + 0.640838 w_k1_DIS_s115
                                 + 0.84151 w_k1_FDX_s115
                                 + 0.564453 w_k1_IBM_s115
                                 + 0.827114 w_k1_INTC_s115
                                 + 0.665196 w_k1_KO_s115
                                 + 0.451759 w_k1_MAT_s115
                                 + 0.790328 w_k1_MSFT_s115
                                 + 0.809268 w_k1_PG_s115 >= 1
 z_k1_s1153_constraint:          var_1 + z_k1_s1153 + 1.29996 w_k1_ADM_s115
                                 + 1.28517 w_k1_AXP_s115
                                 + 1.34558 w_k1_DIS_s115 + 1.1907 w_k1_FDX_s115
                                 + 1.45877 w_k1_IBM_s115
                                 + 1.03456 w_k1_INTC_s115
                                 + 1.17689 w_k1_KO_s115 + 1.51903 w_k1_MAT_s115
                                 + 1.07491 w_k1_MSFT_s115
                                 + 1.22633 w_k1_PG_s115 >= 1
 z_k1_s1154_constraint:          var_1 + z_k1_s1154 + 0.852426 w_k1_ADM_s115
                                 + 0.727741 w_k1_AXP_s115
                                 + 0.996865 w_k1_DIS_s115
                                 + 0.834736 w_k1_FDX_s115
                                 + 1.21538 w_k1_IBM_s115
                                 + 0.933939 w_k1_INTC_s115
                                 + 1.0272 w_k1_KO_s115 + 0.787974 w_k1_MAT_s115
                                 + 1.00796 w_k1_MSFT_s115
                                 + 0.968926 w_k1_PG_s115 >= 1
 z_k1_s1155_constraint:          var_1 + z_k1_s1155 + 0.801378 w_k1_ADM_s115
                                 + 0.648374 w_k1_AXP_s115
                                 + 0.617113 w_k1_DIS_s115
                                 + 0.725589 w_k1_FDX_s115
                                 + 1.01229 w_k1_IBM_s115
                                 + 0.913823 w_k1_INTC_s115
                                 + 0.820889 w_k1_KO_s115
                                 + 0.725173 w_k1_MAT_s115
                                 + 0.846234 w_k1_MSFT_s115
                                 + 0.923932 w_k1_PG_s115 >= 1
 z_k1_s1156_constraint:          var_1 + z_k1_s1156 + 0.810189 w_k1_ADM_s115
                                 + 0.750749 w_k1_AXP_s115
                                 + 0.802907 w_k1_DIS_s115
                                 + 0.837573 w_k1_FDX_s115
                                 + 0.85809 w_k1_IBM_s115
                                 + 0.824019 w_k1_INTC_s115
                                 + 0.784788 w_k1_KO_s115
                                 + 0.865949 w_k1_MAT_s115
                                 + 0.866359 w_k1_MSFT_s115
                                 + 0.967581 w_k1_PG_s115 >= 1
 z_k1_s1157_constraint:          var_1 + z_k1_s1157 + 1.148 w_k1_ADM_s115
                                 + 1.18209 w_k1_AXP_s115
                                 + 1.20498 w_k1_DIS_s115
                                 + 1.21482 w_k1_FDX_s115
                                 + 0.809113 w_k1_IBM_s115
                                 + 1.25237 w_k1_INTC_s115
                                 + 0.943373 w_k1_KO_s115
                                 + 0.677397 w_k1_MAT_s115
                                 + 1.09667 w_k1_MSFT_s115
                                 + 1.19001 w_k1_PG_s115 >= 1
 z_k1_s1160_constraint:          var_1 + z_k1_s1160 + 1.1913 w_k1_ADM_s116
                                 + 1.40203 w_k1_AXP_s116
                                 + 1.77393 w_k1_DIS_s116
                                 + 1.35279 w_k1_FDX_s116
                                 + 1.36057 w_k1_IBM_s116
                                 + 1.34867 w_k1_INTC_s116
                                 + 1.58888 w_k1_KO_s116 + 1.60754 w_k1_MAT_s116
                                 + 1.08769 w_k1_MSFT_s116
                                 + 1.17964 w_k1_PG_s116 >= 1
 z_k1_s1161_constraint:          var_1 + z_k1_s1161 + 1.04303 w_k1_ADM_s116
                                 + 1.00905 w_k1_AXP_s116
                                 + 1.47867 w_k1_DIS_s116
                                 + 1.28044 w_k1_FDX_s116
                                 + 0.982439 w_k1_IBM_s116
                                 + 1.3205 w_k1_INTC_s116 + 1.03849 w_k1_KO_s116
                                 + 0.954702 w_k1_MAT_s116
                                 + 1.37293 w_k1_MSFT_s116
                                 + 1.12463 w_k1_PG_s116 >= 1
 z_k1_s1162_constraint:          var_1 + z_k1_s1162 + 1.14495 w_k1_ADM_s116
                                 + 1.34359 w_k1_AXP_s116
                                 + 1.64088 w_k1_DIS_s116
                                 + 1.26851 w_k1_FDX_s116
                                 + 1.59318 w_k1_IBM_s116
                                 + 1.40459 w_k1_INTC_s116
                                 + 1.49735 w_k1_KO_s116 + 1.84728 w_k1_MAT_s116
                                 + 1.26609 w_k1_MSFT_s116
                                 + 1.28416 w_k1_PG_s116 >= 1
 z_k1_s1163_constraint:          var_1 + z_k1_s1163 + 0.982178 w_k1_ADM_s116
                                 + 0.635308 w_k1_AXP_s116
                                 + 0.757583 w_k1_DIS_s116
                                 + 0.806724 w_k1_FDX_s116
                                 + 0.68278 w_k1_IBM_s116
                                 + 0.839278 w_k1_INTC_s116
                                 + 0.831306 w_k1_KO_s116
                                 + 0.645008 w_k1_MAT_s116
                                 + 0.830069 w_k1_MSFT_s116
                                 + 0.945909 w_k1_PG_s116 >= 1
 z_k1_s1164_constraint:          var_1 + z_k1_s1164 + 1.28509 w_k1_ADM_s116
                                 + 1.17836 w_k1_AXP_s116 + 1.3306 w_k1_DIS_s116
                                 + 1.34317 w_k1_FDX_s116
                                 + 0.895378 w_k1_IBM_s116
                                 + 1.49247 w_k1_INTC_s116
                                 + 1.07454 w_k1_KO_s116 + 0.89275 w_k1_MAT_s116
                                 + 1.14485 w_k1_MSFT_s116 + 1.1495 w_k1_PG_s116
                                 >= 1
 z_k1_s1165_constraint:          var_1 + z_k1_s1165 + 0.783756 w_k1_ADM_s116
                                 + 1.00546 w_k1_AXP_s116
                                 + 0.777848 w_k1_DIS_s116
                                 + 0.847333 w_k1_FDX_s116
                                 + 1.03881 w_k1_IBM_s116
                                 + 1.01088 w_k1_INTC_s116
                                 + 0.829464 w_k1_KO_s116
                                 + 1.06561 w_k1_MAT_s116
                                 + 0.807918 w_k1_MSFT_s116
                                 + 0.863913 w_k1_PG_s116 >= 1
 z_k1_s1166_constraint:          var_1 + z_k1_s1166 + 0.913551 w_k1_ADM_s116
                                 + 0.859576 w_k1_AXP_s116
                                 + 1.06806 w_k1_DIS_s116
                                 + 0.956802 w_k1_FDX_s116
                                 + 1.03181 w_k1_IBM_s116
                                 + 1.14993 w_k1_INTC_s116
                                 + 1.06813 w_k1_KO_s116
                                 + 0.952643 w_k1_MAT_s116
                                 + 1.14134 w_k1_MSFT_s116
                                 + 1.01954 w_k1_PG_s116 >= 1
 z_k1_s1167_constraint:          var_1 + z_k1_s1167 + 1.44989 w_k1_ADM_s116
                                 + 1.68859 w_k1_AXP_s116
                                 + 1.26211 w_k1_DIS_s116
                                 + 1.46199 w_k1_FDX_s116
                                 + 0.986938 w_k1_IBM_s116
                                 + 1.22908 w_k1_INTC_s116
                                 + 1.18282 w_k1_KO_s116 + 1.25465 w_k1_MAT_s116
                                 + 1.12529 w_k1_MSFT_s116
                                 + 1.17723 w_k1_PG_s116 >= 1
 z_k1_s1170_constraint:          var_1 + z_k1_s1170 + 1.23506 w_k1_ADM_s117
                                 + 1.40725 w_k1_AXP_s117
                                 + 1.25523 w_k1_DIS_s117
                                 + 1.22761 w_k1_FDX_s117
                                 + 1.30715 w_k1_IBM_s117
                                 + 0.977026 w_k1_INTC_s117
                                 + 1.25163 w_k1_KO_s117 + 1.43667 w_k1_MAT_s117
                                 + 1.02072 w_k1_MSFT_s117
                                 + 1.08528 w_k1_PG_s117 >= 1
 z_k1_s1171_constraint:          var_1 + z_k1_s1171 + 1.14438 w_k1_ADM_s117
                                 + 0.916697 w_k1_AXP_s117
                                 + 0.761397 w_k1_DIS_s117
                                 + 0.913567 w_k1_FDX_s117
                                 + 0.866894 w_k1_IBM_s117
                                 + 0.952322 w_k1_INTC_s117
                                 + 0.839551 w_k1_KO_s117
                                 + 0.789382 w_k1_MAT_s117
                                 + 0.928538 w_k1_MSFT_s117
                                 + 0.891598 w_k1_PG_s117 >= 1
 z_k1_s1172_constraint:          var_1 + z_k1_s1172 + 0.813671 w_k1_ADM_s117
                                 + 0.831238 w_k1_AXP_s117
                                 + 0.809077 w_k1_DIS_s117
                                 + 0.797259 w_k1_FDX_s117
                                 + 1.18837 w_k1_IBM_s117
                                 + 0.842328 w_k1_INTC_s117
                                 + 1.09616 w_k1_KO_s117 + 1.08422 w_k1_MAT_s117
                                 + 0.867498 w_k1_MSFT_s117
                                 + 0.897621 w_k1_PG_s117 >= 1
 z_k1_s1173_constraint:          var_1 + z_k1_s1173 + 1.05927 w_k1_ADM_s117
                                 + 0.823233 w_k1_AXP_s117
                                 + 0.927884 w_k1_DIS_s117
                                 + 0.868466 w_k1_FDX_s117
                                 + 1.5752 w_k1_IBM_s117
                                 + 0.872522 w_k1_INTC_s117
                                 + 1.09585 w_k1_KO_s117 + 1.05914 w_k1_MAT_s117
                                 + 0.836184 w_k1_MSFT_s117
                                 + 0.868956 w_k1_PG_s117 >= 1
 z_k1_s1174_constraint:          var_1 + z_k1_s1174 + 1.21713 w_k1_ADM_s117
                                 + 1.29299 w_k1_AXP_s117
                                 + 1.78047 w_k1_DIS_s117
                                 + 1.39407 w_k1_FDX_s117
                                 + 1.38577 w_k1_IBM_s117
                                 + 1.86986 w_k1_INTC_s117 + 1.6015 w_k1_KO_s117
                                 + 1.28729 w_k1_MAT_s117
                                 + 1.62696 w_k1_MSFT_s117
                                 + 1.11968 w_k1_PG_s117 >= 1
 z_k1_s1175_constraint:          var_1 + z_k1_s1175 + 1.2154 w_k1_ADM_s117
                                 + 1.02417 w_k1_AXP_s117
                                 + 1.30228 w_k1_DIS_s117
                                 + 1.09199 w_k1_FDX_s117 + 1.1725 w_k1_IBM_s117
                                 + 1.11995 w_k1_INTC_s117
                                 + 1.32878 w_k1_KO_s117
                                 + 0.874798 w_k1_MAT_s117
                                 + 1.00368 w_k1_MSFT_s117
                                 + 1.08477 w_k1_PG_s117 >= 1
 z_k1_s1176_constraint:          var_1 + z_k1_s1176 + 0.74665 w_k1_ADM_s117
                                 + 0.657072 w_k1_AXP_s117
                                 + 0.590759 w_k1_DIS_s117
                                 + 0.750047 w_k1_FDX_s117
                                 + 0.739091 w_k1_IBM_s117
                                 + 0.871201 w_k1_INTC_s117
                                 + 0.699984 w_k1_KO_s117
                                 + 0.483674 w_k1_MAT_s117
                                 + 0.790492 w_k1_MSFT_s117
                                 + 0.830669 w_k1_PG_s117 >= 1
 z_k1_s1177_constraint:          var_1 + z_k1_s1177 + 1.08152 w_k1_ADM_s117
                                 + 1.02636 w_k1_AXP_s117
                                 + 1.11832 w_k1_DIS_s117
                                 + 1.23523 w_k1_FDX_s117
                                 + 0.79723 w_k1_IBM_s117
                                 + 1.12993 w_k1_INTC_s117
                                 + 1.17659 w_k1_KO_s117
                                 + 0.760725 w_k1_MAT_s117
                                 + 1.23339 w_k1_MSFT_s117
                                 + 1.09347 w_k1_PG_s117 >= 1
 k1_return_constraint:           0.0100857116812313 w_k1_ADM_s000
                                 + 0.0228697538229716 w_k1_ADM_s001
                                 + 0.0234213499743518 w_k1_ADM_s002
                                 + 0.00735896022147668 w_k1_ADM_s003
                                 + 0.0248952956508252 w_k1_ADM_s004
                                 + 0.0114381411694124 w_k1_ADM_s005
                                 + 0.00547331753196664 w_k1_ADM_s006
                                 + 0.0143701977628318 w_k1_ADM_s007
                                 + 0.0149577884215243 w_k1_ADM_s010
                                 + 0.00880989739603795 w_k1_ADM_s011
                                 + 0.0134180511175603 w_k1_ADM_s012
                                 + 0.0189455135905143 w_k1_ADM_s013
                                 + 0.0357908178757194 w_k1_ADM_s014
                                 + 0.0189183675439442 w_k1_ADM_s015
                                 + 0.0173457988947713 w_k1_ADM_s016
                                 + 0.00763449534132453 w_k1_ADM_s017
                                 + 0.0331329699856457 w_k1_ADM_s100
                                 + 0.0146316326358699 w_k1_ADM_s101
                                 + 0.0450159360925348 w_k1_ADM_s102
                                 + 0.0606432231882984 w_k1_ADM_s103
                                 + 0.0262569712016534 w_k1_ADM_s104
                                 + 0.0190634622676639 w_k1_ADM_s105
                                 + 0.0511355379314167 w_k1_ADM_s106
                                 + 0.0324641975777561 w_k1_ADM_s107
                                 + 0.0595732359513386 w_k1_ADM_s110
                                 + 0.0971098021787333 w_k1_ADM_s111
                                 + 0.0910041572030519 w_k1_ADM_s112
                                 + 0.029501193171724 w_k1_ADM_s113
                                 + 0.0755393743702415 w_k1_ADM_s114
                                 + 0.0322644408512364 w_k1_ADM_s115
                                 + 0.0227376766223566 w_k1_ADM_s116
                                 + 0.0503710924247838 w_k1_ADM_s117
                                 + 0.00972197192457175 w_k1_AXP_s000
                                 + 0.0273824188976723 w_k1_AXP_s001
                                 + 0.0221101734922529 w_k1_AXP_s002
                                 + 0.00744586116057096 w_k1_AXP_s003
                                 + 0.027908392278277 w_k1_AXP_s004
                                 + 0.0134732706271937 w_k1_AXP_s005
                                 + 0.00482627595342741 w_k1_AXP_s006
                                 + 0.013539611130507 w_k1_AXP_s007
                                 + 0.0158102514641638 w_k1_AXP_s010
                                 + 0.00999672546967723 w_k1_AXP_s011
                                 + 0.0119697680750099 w_k1_AXP_s012
                                 + 0.0232516132765414 w_k1_AXP_s013
                                 + 0.0324301793013437 w_k1_AXP_s014
                                 + 0.0246346514845878 w_k1_AXP_s015
                                 + 0.0180053932007121 w_k1_AXP_s016
                                 + 0.00861896703456593 w_k1_AXP_s017
                                 + 0.0350582233712703 w_k1_AXP_s100
                                 + 0.0176706831415341 w_k1_AXP_s101
                                 + 0.0545448371059047 w_k1_AXP_s102
                                 + 0.0631714741107116 w_k1_AXP_s103
                                 + 0.0289357253018253 w_k1_AXP_s104
                                 + 0.0214846891430403 w_k1_AXP_s105
                                 + 0.0638893102436791 w_k1_AXP_s106
                                 + 0.0282237620720187 w_k1_AXP_s107
                                 + 0.0622890498084035 w_k1_AXP_s110
                                 + 0.0859437906074304 w_k1_AXP_s111
                                 + 0.0997727503258516 w_k1_AXP_s112
                                 + 0.0294032294395098 w_k1_AXP_s113
                                 + 0.0730248334487817 w_k1_AXP_s114
                                 + 0.0291145210226033 w_k1_AXP_s115
                                 + 0.0229174152119304 w_k1_AXP_s116
                                 + 0.0466200471574172 w_k1_AXP_s117
                                 + 0.01046994890633 w_k1_DIS_s000
                                 + 0.0248448732058669 w_k1_DIS_s001
                                 + 0.0243980816070384 w_k1_DIS_s002
                                 + 0.00804113705809979 w_k1_DIS_s003
                                 + 0.0259212087999322 w_k1_DIS_s004
                                 + 0.0117680259701169 w_k1_DIS_s005
                                 + 0.00564794631516296 w_k1_DIS_s006
                                 + 0.0144957443692161 w_k1_DIS_s007
                                 + 0.015692358243115 w_k1_DIS_s010
                                 + 0.00925421925324344 w_k1_DIS_s011
                                 + 0.0132025568186778 w_k1_DIS_s012
                                 + 0.0232105947462428 w_k1_DIS_s013
                                 + 0.0317093291435174 w_k1_DIS_s014
                                 + 0.0246019274214473 w_k1_DIS_s015
                                 + 0.0184670334787943 w_k1_DIS_s016
                                 + 0.00843103279051508 w_k1_DIS_s017
                                 + 0.0356404159546338 w_k1_DIS_s100
                                 + 0.0165498704612685 w_k1_DIS_s101
                                 + 0.0564431070742197 w_k1_DIS_s102
                                 + 0.068128176523579 w_k1_DIS_s103
                                 + 0.0284727974054343 w_k1_DIS_s104
                                 + 0.0228733554943733 w_k1_DIS_s105
                                 + 0.0607258764634568 w_k1_DIS_s106
                                 + 0.0291066770497331 w_k1_DIS_s107
                                 + 0.0601468562570017 w_k1_DIS_s110
                                 + 0.0911778496787245 w_k1_DIS_s111
                                 + 0.0995104862564264 w_k1_DIS_s112
                                 + 0.0281534569840707 w_k1_DIS_s113
                                 + 0.0756221604897498 w_k1_DIS_s114
                                 + 0.0320397909838663 w_k1_DIS_s115
                                 + 0.0250063284314286 w_k1_DIS_s116
                                 + 0.0473257647085107 w_k1_DIS_s117
                                 + 0.0102309828441567 w_k1_FDX_s000
                                 + 0.0261346200272686 w_k1_FDX_s001
                                 + 0.0216938958157318 w_k1_FDX_s002
                                 + 0.00724397690156608 w_k1_FDX_s003
                                 + 0.026223176497832 w_k1_FDX_s004
                                 + 0.0115561533040999 w_k1_FDX_s005
                                 + 0.00530848508443241 w_k1_FDX_s006
                                 + 0.0147450291395963 w_k1_FDX_s007
                                 + 0.0159996513962589 w_k1_FDX_s010
                                 + 0.00924896787760935 w_k1_FDX_s011
                                 + 0.0128497295899292 w_k1_FDX_s012
                                 + 0.0227977982673696 w_k1_FDX_s013
                                 + 0.0323656990543973 w_k1_FDX_s014
                                 + 0.0237427451116473 w_k1_FDX_s015
                                 + 0.0164628886781396 w_k1_FDX_s016
                                 + 0.00812587756125008 w_k1_FDX_s017
                                 + 0.0356227142871526 w_k1_FDX_s100
                                 + 0.0162220672099899 w_k1_FDX_s101
                                 + 0.0550533568587231 w_k1_FDX_s102
                                 + 0.0612899011362097 w_k1_FDX_s103
                                 + 0.0263019632546158 w_k1_FDX_s104
                                 + 0.0224943750702797 w_k1_FDX_s105
                                 + 0.0586455571015785 w_k1_FDX_s106
                                 + 0.0293905789312377 w_k1_FDX_s107
                                 + 0.0583147431642836 w_k1_FDX_s110
                                 + 0.0852545795492439 w_k1_FDX_s111
                                 + 0.0946923066365007 w_k1_FDX_s112
                                 + 0.0285251505163234 w_k1_FDX_s113
                                 + 0.0751997943723602 w_k1_FDX_s114
                                 + 0.0309105983528872 w_k1_FDX_s115
                                 + 0.0236400162878453 w_k1_FDX_s116
                                 + 0.0477779047623969 w_k1_FDX_s117
                                 + 0.0105293865124829 w_k1_IBM_s000
                                 + 0.0247692288367637 w_k1_IBM_s001
                                 + 0.0218056062957507 w_k1_IBM_s002
                                 + 0.00740361964373898 w_k1_IBM_s003
                                 + 0.0271839005653877 w_k1_IBM_s004
                                 + 0.0123954964974898 w_k1_IBM_s005
                                 + 0.00534819240682247 w_k1_IBM_s006
                                 + 0.0150586580056028 w_k1_IBM_s007
                                 + 0.0155739620677706 w_k1_IBM_s010
                                 + 0.00949401996078921 w_k1_IBM_s011
                                 + 0.012756459310819 w_k1_IBM_s012
                                 + 0.023008185742351 w_k1_IBM_s013
                                 + 0.0308524375285944 w_k1_IBM_s014
                                 + 0.0239270581937864 w_k1_IBM_s015
                                 + 0.0174163695854887 w_k1_IBM_s016
                                 + 0.00804298908717006 w_k1_IBM_s017
                                 + 0.0369477137676893 w_k1_IBM_s100
                                 + 0.0166762749468312 w_k1_IBM_s101
                                 + 0.0540665075295518 w_k1_IBM_s102
                                 + 0.0633209844340077 w_k1_IBM_s103
                                 + 0.0290947862456748 w_k1_IBM_s104
                                 + 0.0237138639723516 w_k1_IBM_s105
                                 + 0.0591233123826928 w_k1_IBM_s106
                                 + 0.0288910427119811 w_k1_IBM_s107
                                 + 0.060417646474237 w_k1_IBM_s110
                                 + 0.0844583112052043 w_k1_IBM_s111
                                 + 0.0996723880256384 w_k1_IBM_s112
                                 + 0.0284965148041853 w_k1_IBM_s113
                                 + 0.0707457133109173 w_k1_IBM_s114
                                 + 0.030749211646746 w_k1_IBM_s115
                                 + 0.0228007778097894 w_k1_IBM_s116
                                 + 0.0501729338156627 w_k1_IBM_s117
                                 + 0.00947453402545433 w_k1_INTC_s000
                                 + 0.0259333462078391 w_k1_INTC_s001
                                 + 0.0198037973579416 w_k1_INTC_s002
                                 + 0.00713187404515202 w_k1_INTC_s003
                                 + 0.0274603984375464 w_k1_INTC_s004
                                 + 0.0116329215664434 w_k1_INTC_s005
                                 + 0.00452087823470452 w_k1_INTC_s006
                                 + 0.0161943678689485 w_k1_INTC_s007
                                 + 0.0173900458800294 w_k1_INTC_s010
                                 + 0.00906994739105253 w_k1_INTC_s011
                                 + 0.0107500984720461 w_k1_INTC_s012
                                 + 0.0227875568730296 w_k1_INTC_s013
                                 + 0.0325259269666382 w_k1_INTC_s014
                                 + 0.0238749533826458 w_k1_INTC_s015
                                 + 0.015978067193808 w_k1_INTC_s016
                                 + 0.0073028897081715 w_k1_INTC_s017
                                 + 0.0345141556462941 w_k1_INTC_s100
                                 + 0.0162533953855406 w_k1_INTC_s101
                                 + 0.0595009173584557 w_k1_INTC_s102
                                 + 0.0653371135388 w_k1_INTC_s103
                                 + 0.0319902558082205 w_k1_INTC_s104
                                 + 0.0224522261786212 w_k1_INTC_s105
                                 + 0.0632983696963427 w_k1_INTC_s106
                                 + 0.0306302387366363 w_k1_INTC_s107
                                 + 0.057475390031935 w_k1_INTC_s110
                                 + 0.0910519624352493 w_k1_INTC_s111
                                 + 0.0970133751257883 w_k1_INTC_s112
                                 + 0.0266211796764229 w_k1_INTC_s113
                                 + 0.0687507666140219 w_k1_INTC_s114
                                 + 0.030413828504204 w_k1_INTC_s115
                                 + 0.0257442059864472 w_k1_INTC_s116
                                 + 0.048112521850497 w_k1_INTC_s117
                                 + 0.0107862954989244 w_k1_KO_s000
                                 + 0.0226393354588511 w_k1_KO_s001
                                 + 0.0199159367982497 w_k1_KO_s002
                                 + 0.00806461595139037 w_k1_KO_s003
                                 + 0.0251993520632199 w_k1_KO_s004
                                 + 0.0111734132542305 w_k1_KO_s005
                                 + 0.00574056418478987 w_k1_KO_s006
                                 + 0.0173561189320214 w_k1_KO_s007
                                 + 0.0160397005093011 w_k1_KO_s010
                                 + 0.00827925559422632 w_k1_KO_s011
                                 + 0.0130258060157724 w_k1_KO_s012
                                 + 0.0228049613528225 w_k1_KO_s013
                                 + 0.0296843559107342 w_k1_KO_s014
                                 + 0.0240984612083676 w_k1_KO_s015
                                 + 0.0169170387032946 w_k1_KO_s016
                                 + 0.00780310337359311 w_k1_KO_s017
                                 + 0.0396081211427144 w_k1_KO_s100
                                 + 0.0168111153720988 w_k1_KO_s101
                                 + 0.052299929713874 w_k1_KO_s102
                                 + 0.0630014284404729 w_k1_KO_s103
                                 + 0.0303284506668391 w_k1_KO_s104
                                 + 0.0227014841811645 w_k1_KO_s105
                                 + 0.0647393293153938 w_k1_KO_s106
                                 + 0.0313932678852558 w_k1_KO_s107
                                 + 0.0626490461872198 w_k1_KO_s110
                                 + 0.0857577832896494 w_k1_KO_s111
                                 + 0.0982825583576332 w_k1_KO_s112
                                 + 0.03017600155674 w_k1_KO_s113
                                 + 0.0767533323172958 w_k1_KO_s114
                                 + 0.0286110120221575 w_k1_KO_s115
                                 + 0.0236019990261355 w_k1_KO_s116
                                 + 0.0510623813962028 w_k1_KO_s117
                                 + 0.0123716839982807 w_k1_MAT_s000
                                 + 0.0284121696102868 w_k1_MAT_s001
                                 + 0.0177868670378472 w_k1_MAT_s002
                                 + 0.00750015649913982 w_k1_MAT_s003
                                 + 0.0325412156180681 w_k1_MAT_s004
                                 + 0.012933459474116 w_k1_MAT_s005
                                 + 0.00568631016983577 w_k1_MAT_s006
                                 + 0.0179342498993403 w_k1_MAT_s007
                                 + 0.0154853471093002 w_k1_MAT_s010
                                 + 0.0106082494471836 w_k1_MAT_s011
                                 + 0.0130994112562237 w_k1_MAT_s012
                                 + 0.0237839772880204 w_k1_MAT_s013
                                 + 0.0316213495126466 w_k1_MAT_s014
                                 + 0.0247121628582906 w_k1_MAT_s015
                                 + 0.0195483486466244 w_k1_MAT_s016
                                 + 0.0103777856533701 w_k1_MAT_s017
                                 + 0.0465542727791056 w_k1_MAT_s100
                                 + 0.0157577880296985 w_k1_MAT_s101
                                 + 0.055183906295408 w_k1_MAT_s102
                                 + 0.0723437266171795 w_k1_MAT_s103
                                 + 0.0267606608520008 w_k1_MAT_s104
                                 + 0.0233838606419076 w_k1_MAT_s105
                                 + 0.0562656818665782 w_k1_MAT_s106
                                 + 0.0294238981447541 w_k1_MAT_s107
                                 + 0.0599424726182793 w_k1_MAT_s110
                                 + 0.0888523577723135 w_k1_MAT_s111
                                 + 0.0913420744439026 w_k1_MAT_s112
                                 + 0.0300126556053632 w_k1_MAT_s113
                                 + 0.0783829776547588 w_k1_MAT_s114
                                 + 0.026002965534436 w_k1_MAT_s115
                                 + 0.0234617103750909 w_k1_MAT_s116
                                 + 0.0441912026152983 w_k1_MAT_s117
                                 + 0.0104690852361391 w_k1_MSFT_s000
                                 + 0.0253717007132141 w_k1_MSFT_s001
                                 + 0.0207317995317675 w_k1_MSFT_s002
                                 + 0.00890571410981732 w_k1_MSFT_s003
                                 + 0.024860776321538 w_k1_MSFT_s004
                                 + 0.0107147254691555 w_k1_MSFT_s005
                                 + 0.00530860547190958 w_k1_MSFT_s006
                                 + 0.0160624389206662 w_k1_MSFT_s007
                                 + 0.0145908882205605 w_k1_MSFT_s010
                                 + 0.00848821367422745 w_k1_MSFT_s011
                                 + 0.0131463801928125 w_k1_MSFT_s012
                                 + 0.0231351601523302 w_k1_MSFT_s013
                                 + 0.0315230687428648 w_k1_MSFT_s014
                                 + 0.022099278705897 w_k1_MSFT_s015
                                 + 0.0165474514032655 w_k1_MSFT_s016
                                 + 0.00789398316753932 w_k1_MSFT_s017
                                 + 0.0363879290280552 w_k1_MSFT_s100
                                 + 0.016130088189042 w_k1_MSFT_s101
                                 + 0.058751768263435 w_k1_MSFT_s102
                                 + 0.0663139611545346 w_k1_MSFT_s103
                                 + 0.0290091290631942 w_k1_MSFT_s104
                                 + 0.0228327856591303 w_k1_MSFT_s105
                                 + 0.0594319497981537 w_k1_MSFT_s106
                                 + 0.0300619630606983 w_k1_MSFT_s107
                                 + 0.0584502244152285 w_k1_MSFT_s110
                                 + 0.0840826797373412 w_k1_MSFT_s111
                                 + 0.0929609947524451 w_k1_MSFT_s112
                                 + 0.0297181344041773 w_k1_MSFT_s113
                                 + 0.075939292312842 w_k1_MSFT_s114
                                 + 0.0301553295165268 w_k1_MSFT_s115
                                 + 0.0234949294447977 w_k1_MSFT_s116
                                 + 0.0470509737091961 w_k1_MSFT_s117
                                 + 0.0106184548938409 w_k1_PG_s000
                                 + 0.0244189406516021 w_k1_PG_s001
                                 + 0.0204423654697924 w_k1_PG_s002
                                 + 0.00747809559905963 w_k1_PG_s003
                                 + 0.0251914696597983 w_k1_PG_s004
                                 + 0.011542690296493 w_k1_PG_s005
                                 + 0.00523715389494626 w_k1_PG_s006
                                 + 0.0146025499217453 w_k1_PG_s007
                                 + 0.0152122494120839 w_k1_PG_s010
                                 + 0.00933774826515098 w_k1_PG_s011
                                 + 0.0120975891104099 w_k1_PG_s012
                                 + 0.0209541065282928 w_k1_PG_s013
                                 + 0.0319804643392604 w_k1_PG_s014
                                 + 0.023656864596613 w_k1_PG_s015
                                 + 0.0172567413387958 w_k1_PG_s016
                                 + 0.00800554734017984 w_k1_PG_s017
                                 + 0.0342782293408928 w_k1_PG_s100
                                 + 0.0151045879720922 w_k1_PG_s101
                                 + 0.0576371720289173 w_k1_PG_s102
                                 + 0.063949534663661 w_k1_PG_s103
                                 + 0.0268287812094314 w_k1_PG_s104
                                 + 0.0221694869935589 w_k1_PG_s105
                                 + 0.0575271312086724 w_k1_PG_s106
                                 + 0.027788591235214 w_k1_PG_s107
                                 + 0.0636920269359644 w_k1_PG_s110
                                 + 0.0812534399943978 w_k1_PG_s111
                                 + 0.0914358559541836 w_k1_PG_s112
                                 + 0.0283646377003679 w_k1_PG_s113
                                 + 0.0764890466553159 w_k1_PG_s114
                                 + 0.0320010306099406 w_k1_PG_s115
                                 + 0.0232276407657712 w_k1_PG_s116
                                 + 0.0460063048695431 w_k1_PG_s117 >= 1.19
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
