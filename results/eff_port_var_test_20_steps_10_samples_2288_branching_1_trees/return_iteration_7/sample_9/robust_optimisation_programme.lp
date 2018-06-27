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
 w_k1_ADM_s0_return_constraint:  1.09947 w_0_ADM - w_k1_ADM_s0
                                 + 0.99 b_k1_ADM_s0 - 1.01 s_k1_ADM_s0  = 0
 w_k1_ADM_s1_return_constraint:  0.936626 w_0_ADM - w_k1_ADM_s1
                                 + 0.99 b_k1_ADM_s1 - 1.01 s_k1_ADM_s1  = 0
 w_k1_ADM_s00_return_constraint: 1.08511 w_k1_ADM_s0 - w_k1_ADM_s00
                                 + 0.99 b_k1_ADM_s00 - 1.01 s_k1_ADM_s00  = 0
 w_k1_ADM_s01_return_constraint: 0.847235 w_k1_ADM_s0 - w_k1_ADM_s01
                                 + 0.99 b_k1_ADM_s01 - 1.01 s_k1_ADM_s01  = 0
 w_k1_ADM_s10_return_constraint: 1.06126 w_k1_ADM_s1 - w_k1_ADM_s10
                                 + 0.99 b_k1_ADM_s10 - 1.01 s_k1_ADM_s10  = 0
 w_k1_ADM_s11_return_constraint: 0.881627 w_k1_ADM_s1 - w_k1_ADM_s11
                                 + 0.99 b_k1_ADM_s11 - 1.01 s_k1_ADM_s11  = 0
 w_k1_ADM_s000_return_constraint: 1.06839 w_k1_ADM_s00 - w_k1_ADM_s000
                                 + 0.99 b_k1_ADM_s000 - 1.01 s_k1_ADM_s000  = 0
 w_k1_ADM_s001_return_constraint: 1.21529 w_k1_ADM_s00 - w_k1_ADM_s001
                                 + 0.99 b_k1_ADM_s001 - 1.01 s_k1_ADM_s001  = 0
 w_k1_ADM_s002_return_constraint: 0.851408 w_k1_ADM_s00 - w_k1_ADM_s002
                                 + 0.99 b_k1_ADM_s002 - 1.01 s_k1_ADM_s002  = 0
 w_k1_ADM_s003_return_constraint: 1.12818 w_k1_ADM_s00 - w_k1_ADM_s003
                                 + 0.99 b_k1_ADM_s003 - 1.01 s_k1_ADM_s003  = 0
 w_k1_ADM_s004_return_constraint: 0.98811 w_k1_ADM_s00 - w_k1_ADM_s004
                                 + 0.99 b_k1_ADM_s004 - 1.01 s_k1_ADM_s004  = 0
 w_k1_ADM_s005_return_constraint: 0.987243 w_k1_ADM_s00 - w_k1_ADM_s005
                                 + 0.99 b_k1_ADM_s005 - 1.01 s_k1_ADM_s005  = 0
 w_k1_ADM_s006_return_constraint: 0.953662 w_k1_ADM_s00 - w_k1_ADM_s006
                                 + 0.99 b_k1_ADM_s006 - 1.01 s_k1_ADM_s006  = 0
 w_k1_ADM_s007_return_constraint: 1.45996 w_k1_ADM_s00 - w_k1_ADM_s007
                                 + 0.99 b_k1_ADM_s007 - 1.01 s_k1_ADM_s007  = 0
 w_k1_ADM_s010_return_constraint: 0.843491 w_k1_ADM_s01 - w_k1_ADM_s010
                                 + 0.99 b_k1_ADM_s010 - 1.01 s_k1_ADM_s010  = 0
 w_k1_ADM_s011_return_constraint: 0.987139 w_k1_ADM_s01 - w_k1_ADM_s011
                                 + 0.99 b_k1_ADM_s011 - 1.01 s_k1_ADM_s011  = 0
 w_k1_ADM_s012_return_constraint: 1.16028 w_k1_ADM_s01 - w_k1_ADM_s012
                                 + 0.99 b_k1_ADM_s012 - 1.01 s_k1_ADM_s012  = 0
 w_k1_ADM_s013_return_constraint: 0.797756 w_k1_ADM_s01 - w_k1_ADM_s013
                                 + 0.99 b_k1_ADM_s013 - 1.01 s_k1_ADM_s013  = 0
 w_k1_ADM_s014_return_constraint: 0.840319 w_k1_ADM_s01 - w_k1_ADM_s014
                                 + 0.99 b_k1_ADM_s014 - 1.01 s_k1_ADM_s014  = 0
 w_k1_ADM_s015_return_constraint: 1.26158 w_k1_ADM_s01 - w_k1_ADM_s015
                                 + 0.99 b_k1_ADM_s015 - 1.01 s_k1_ADM_s015  = 0
 w_k1_ADM_s016_return_constraint: 1.2164 w_k1_ADM_s01 - w_k1_ADM_s016
                                 + 0.99 b_k1_ADM_s016 - 1.01 s_k1_ADM_s016  = 0
 w_k1_ADM_s017_return_constraint: 1.27364 w_k1_ADM_s01 - w_k1_ADM_s017
                                 + 0.99 b_k1_ADM_s017 - 1.01 s_k1_ADM_s017  = 0
 w_k1_ADM_s100_return_constraint: 0.965431 w_k1_ADM_s10 - w_k1_ADM_s100
                                 + 0.99 b_k1_ADM_s100 - 1.01 s_k1_ADM_s100  = 0
 w_k1_ADM_s101_return_constraint: 1.34096 w_k1_ADM_s10 - w_k1_ADM_s101
                                 + 0.99 b_k1_ADM_s101 - 1.01 s_k1_ADM_s101  = 0
 w_k1_ADM_s102_return_constraint: 1.25683 w_k1_ADM_s10 - w_k1_ADM_s102
                                 + 0.99 b_k1_ADM_s102 - 1.01 s_k1_ADM_s102  = 0
 w_k1_ADM_s103_return_constraint: 1.2442 w_k1_ADM_s10 - w_k1_ADM_s103
                                 + 0.99 b_k1_ADM_s103 - 1.01 s_k1_ADM_s103  = 0
 w_k1_ADM_s104_return_constraint: 1.00377 w_k1_ADM_s10 - w_k1_ADM_s104
                                 + 0.99 b_k1_ADM_s104 - 1.01 s_k1_ADM_s104  = 0
 w_k1_ADM_s105_return_constraint: 0.794586 w_k1_ADM_s10 - w_k1_ADM_s105
                                 + 0.99 b_k1_ADM_s105 - 1.01 s_k1_ADM_s105  = 0
 w_k1_ADM_s106_return_constraint: 0.79066 w_k1_ADM_s10 - w_k1_ADM_s106
                                 + 0.99 b_k1_ADM_s106 - 1.01 s_k1_ADM_s106  = 0
 w_k1_ADM_s107_return_constraint: 0.918202 w_k1_ADM_s10 - w_k1_ADM_s107
                                 + 0.99 b_k1_ADM_s107 - 1.01 s_k1_ADM_s107  = 0
 w_k1_ADM_s110_return_constraint: 0.804492 w_k1_ADM_s11 - w_k1_ADM_s110
                                 + 0.99 b_k1_ADM_s110 - 1.01 s_k1_ADM_s110  = 0
 w_k1_ADM_s111_return_constraint: 1.07106 w_k1_ADM_s11 - w_k1_ADM_s111
                                 + 0.99 b_k1_ADM_s111 - 1.01 s_k1_ADM_s111  = 0
 w_k1_ADM_s112_return_constraint: 0.780152 w_k1_ADM_s11 - w_k1_ADM_s112
                                 + 0.99 b_k1_ADM_s112 - 1.01 s_k1_ADM_s112  = 0
 w_k1_ADM_s113_return_constraint: 1.04796 w_k1_ADM_s11 - w_k1_ADM_s113
                                 + 0.99 b_k1_ADM_s113 - 1.01 s_k1_ADM_s113  = 0
 w_k1_ADM_s114_return_constraint: 1.0349 w_k1_ADM_s11 - w_k1_ADM_s114
                                 + 0.99 b_k1_ADM_s114 - 1.01 s_k1_ADM_s114  = 0
 w_k1_ADM_s115_return_constraint: 1.18086 w_k1_ADM_s11 - w_k1_ADM_s115
                                 + 0.99 b_k1_ADM_s115 - 1.01 s_k1_ADM_s115  = 0
 w_k1_ADM_s116_return_constraint: 0.906512 w_k1_ADM_s11 - w_k1_ADM_s116
                                 + 0.99 b_k1_ADM_s116 - 1.01 s_k1_ADM_s116  = 0
 w_k1_ADM_s117_return_constraint: 0.964994 w_k1_ADM_s11 - w_k1_ADM_s117
                                 + 0.99 b_k1_ADM_s117 - 1.01 s_k1_ADM_s117  = 0
 w_k1_AXP_s0_return_constraint:  1.15973 w_0_AXP - w_k1_AXP_s0
                                 + 0.99 b_k1_AXP_s0 - 1.01 s_k1_AXP_s0  = 0
 w_k1_AXP_s1_return_constraint:  0.965019 w_0_AXP - w_k1_AXP_s1
                                 + 0.99 b_k1_AXP_s1 - 1.01 s_k1_AXP_s1  = 0
 w_k1_AXP_s00_return_constraint: 1.20896 w_k1_AXP_s0 - w_k1_AXP_s00
                                 + 0.99 b_k1_AXP_s00 - 1.01 s_k1_AXP_s00  = 0
 w_k1_AXP_s01_return_constraint: 0.743583 w_k1_AXP_s0 - w_k1_AXP_s01
                                 + 0.99 b_k1_AXP_s01 - 1.01 s_k1_AXP_s01  = 0
 w_k1_AXP_s10_return_constraint: 1.08021 w_k1_AXP_s1 - w_k1_AXP_s10
                                 + 0.99 b_k1_AXP_s10 - 1.01 s_k1_AXP_s10  = 0
 w_k1_AXP_s11_return_constraint: 0.695402 w_k1_AXP_s1 - w_k1_AXP_s11
                                 + 0.99 b_k1_AXP_s11 - 1.01 s_k1_AXP_s11  = 0
 w_k1_AXP_s000_return_constraint: 0.92718 w_k1_AXP_s00 - w_k1_AXP_s000
                                 + 0.99 b_k1_AXP_s000 - 1.01 s_k1_AXP_s000  = 0
 w_k1_AXP_s001_return_constraint: 1.09534 w_k1_AXP_s00 - w_k1_AXP_s001
                                 + 0.99 b_k1_AXP_s001 - 1.01 s_k1_AXP_s001  = 0
 w_k1_AXP_s002_return_constraint: 0.77892 w_k1_AXP_s00 - w_k1_AXP_s002
                                 + 0.99 b_k1_AXP_s002 - 1.01 s_k1_AXP_s002  = 0
 w_k1_AXP_s003_return_constraint: 1.26483 w_k1_AXP_s00 - w_k1_AXP_s003
                                 + 0.99 b_k1_AXP_s003 - 1.01 s_k1_AXP_s003  = 0
 w_k1_AXP_s004_return_constraint: 1.07102 w_k1_AXP_s00 - w_k1_AXP_s004
                                 + 0.99 b_k1_AXP_s004 - 1.01 s_k1_AXP_s004  = 0
 w_k1_AXP_s005_return_constraint: 0.950801 w_k1_AXP_s00 - w_k1_AXP_s005
                                 + 0.99 b_k1_AXP_s005 - 1.01 s_k1_AXP_s005  = 0
 w_k1_AXP_s006_return_constraint: 0.754166 w_k1_AXP_s00 - w_k1_AXP_s006
                                 + 0.99 b_k1_AXP_s006 - 1.01 s_k1_AXP_s006  = 0
 w_k1_AXP_s007_return_constraint: 1.70924 w_k1_AXP_s00 - w_k1_AXP_s007
                                 + 0.99 b_k1_AXP_s007 - 1.01 s_k1_AXP_s007  = 0
 w_k1_AXP_s010_return_constraint: 0.790125 w_k1_AXP_s01 - w_k1_AXP_s010
                                 + 0.99 b_k1_AXP_s010 - 1.01 s_k1_AXP_s010  = 0
 w_k1_AXP_s011_return_constraint: 1.0813 w_k1_AXP_s01 - w_k1_AXP_s011
                                 + 0.99 b_k1_AXP_s011 - 1.01 s_k1_AXP_s011  = 0
 w_k1_AXP_s012_return_constraint: 1.02038 w_k1_AXP_s01 - w_k1_AXP_s012
                                 + 0.99 b_k1_AXP_s012 - 1.01 s_k1_AXP_s012  = 0
 w_k1_AXP_s013_return_constraint: 0.995807 w_k1_AXP_s01 - w_k1_AXP_s013
                                 + 0.99 b_k1_AXP_s013 - 1.01 s_k1_AXP_s013  = 0
 w_k1_AXP_s014_return_constraint: 0.645811 w_k1_AXP_s01 - w_k1_AXP_s014
                                 + 0.99 b_k1_AXP_s014 - 1.01 s_k1_AXP_s014  = 0
 w_k1_AXP_s015_return_constraint: 1.23111 w_k1_AXP_s01 - w_k1_AXP_s015
                                 + 0.99 b_k1_AXP_s015 - 1.01 s_k1_AXP_s015  = 0
 w_k1_AXP_s016_return_constraint: 1.55172 w_k1_AXP_s01 - w_k1_AXP_s016
                                 + 0.99 b_k1_AXP_s016 - 1.01 s_k1_AXP_s016  = 0
 w_k1_AXP_s017_return_constraint: 1.29806 w_k1_AXP_s01 - w_k1_AXP_s017
                                 + 0.99 b_k1_AXP_s017 - 1.01 s_k1_AXP_s017  = 0
 w_k1_AXP_s100_return_constraint: 1.0531 w_k1_AXP_s10 - w_k1_AXP_s100
                                 + 0.99 b_k1_AXP_s100 - 1.01 s_k1_AXP_s100  = 0
 w_k1_AXP_s101_return_constraint: 1.79725 w_k1_AXP_s10 - w_k1_AXP_s101
                                 + 0.99 b_k1_AXP_s101 - 1.01 s_k1_AXP_s101  = 0
 w_k1_AXP_s102_return_constraint: 1.2483 w_k1_AXP_s10 - w_k1_AXP_s102
                                 + 0.99 b_k1_AXP_s102 - 1.01 s_k1_AXP_s102  = 0
 w_k1_AXP_s103_return_constraint: 1.43739 w_k1_AXP_s10 - w_k1_AXP_s103
                                 + 0.99 b_k1_AXP_s103 - 1.01 s_k1_AXP_s103  = 0
 w_k1_AXP_s104_return_constraint: 1.14496 w_k1_AXP_s10 - w_k1_AXP_s104
                                 + 0.99 b_k1_AXP_s104 - 1.01 s_k1_AXP_s104  = 0
 w_k1_AXP_s105_return_constraint: 0.765775 w_k1_AXP_s10 - w_k1_AXP_s105
                                 + 0.99 b_k1_AXP_s105 - 1.01 s_k1_AXP_s105  = 0
 w_k1_AXP_s106_return_constraint: 0.977692 w_k1_AXP_s10 - w_k1_AXP_s106
                                 + 0.99 b_k1_AXP_s106 - 1.01 s_k1_AXP_s106  = 0
 w_k1_AXP_s107_return_constraint: 0.869553 w_k1_AXP_s10 - w_k1_AXP_s107
                                 + 0.99 b_k1_AXP_s107 - 1.01 s_k1_AXP_s107  = 0
 w_k1_AXP_s110_return_constraint: 0.712868 w_k1_AXP_s11 - w_k1_AXP_s110
                                 + 0.99 b_k1_AXP_s110 - 1.01 s_k1_AXP_s110  = 0
 w_k1_AXP_s111_return_constraint: 1.46106 w_k1_AXP_s11 - w_k1_AXP_s111
                                 + 0.99 b_k1_AXP_s111 - 1.01 s_k1_AXP_s111  = 0
 w_k1_AXP_s112_return_constraint: 0.754144 w_k1_AXP_s11 - w_k1_AXP_s112
                                 + 0.99 b_k1_AXP_s112 - 1.01 s_k1_AXP_s112  = 0
 w_k1_AXP_s113_return_constraint: 1.17045 w_k1_AXP_s11 - w_k1_AXP_s113
                                 + 0.99 b_k1_AXP_s113 - 1.01 s_k1_AXP_s113  = 0
 w_k1_AXP_s114_return_constraint: 0.903014 w_k1_AXP_s11 - w_k1_AXP_s114
                                 + 0.99 b_k1_AXP_s114 - 1.01 s_k1_AXP_s114  = 0
 w_k1_AXP_s115_return_constraint: 1.29102 w_k1_AXP_s11 - w_k1_AXP_s115
                                 + 0.99 b_k1_AXP_s115 - 1.01 s_k1_AXP_s115  = 0
 w_k1_AXP_s116_return_constraint: 1.16403 w_k1_AXP_s11 - w_k1_AXP_s116
                                 + 0.99 b_k1_AXP_s116 - 1.01 s_k1_AXP_s116  = 0
 w_k1_AXP_s117_return_constraint: 0.974957 w_k1_AXP_s11 - w_k1_AXP_s117
                                 + 0.99 b_k1_AXP_s117 - 1.01 s_k1_AXP_s117  = 0
 w_k1_DIS_s0_return_constraint:  1.20022 w_0_DIS - w_k1_DIS_s0
                                 + 0.99 b_k1_DIS_s0 - 1.01 s_k1_DIS_s0  = 0
 w_k1_DIS_s1_return_constraint:  0.927106 w_0_DIS - w_k1_DIS_s1
                                 + 0.99 b_k1_DIS_s1 - 1.01 s_k1_DIS_s1  = 0
 w_k1_DIS_s00_return_constraint: 1.35116 w_k1_DIS_s0 - w_k1_DIS_s00
                                 + 0.99 b_k1_DIS_s00 - 1.01 s_k1_DIS_s00  = 0
 w_k1_DIS_s01_return_constraint: 0.758939 w_k1_DIS_s0 - w_k1_DIS_s01
                                 + 0.99 b_k1_DIS_s01 - 1.01 s_k1_DIS_s01  = 0
 w_k1_DIS_s10_return_constraint: 1.2058 w_k1_DIS_s1 - w_k1_DIS_s10
                                 + 0.99 b_k1_DIS_s10 - 1.01 s_k1_DIS_s10  = 0
 w_k1_DIS_s11_return_constraint: 0.678045 w_k1_DIS_s1 - w_k1_DIS_s11
                                 + 0.99 b_k1_DIS_s11 - 1.01 s_k1_DIS_s11  = 0
 w_k1_DIS_s000_return_constraint: 1.00277 w_k1_DIS_s00 - w_k1_DIS_s000
                                 + 0.99 b_k1_DIS_s000 - 1.01 s_k1_DIS_s000  = 0
 w_k1_DIS_s001_return_constraint: 1.22253 w_k1_DIS_s00 - w_k1_DIS_s001
                                 + 0.99 b_k1_DIS_s001 - 1.01 s_k1_DIS_s001  = 0
 w_k1_DIS_s002_return_constraint: 0.693322 w_k1_DIS_s00 - w_k1_DIS_s002
                                 + 0.99 b_k1_DIS_s002 - 1.01 s_k1_DIS_s002  = 0
 w_k1_DIS_s003_return_constraint: 1.29433 w_k1_DIS_s00 - w_k1_DIS_s003
                                 + 0.99 b_k1_DIS_s003 - 1.01 s_k1_DIS_s003  = 0
 w_k1_DIS_s004_return_constraint: 0.78723 w_k1_DIS_s00 - w_k1_DIS_s004
                                 + 0.99 b_k1_DIS_s004 - 1.01 s_k1_DIS_s004  = 0
 w_k1_DIS_s005_return_constraint: 0.665936 w_k1_DIS_s00 - w_k1_DIS_s005
                                 + 0.99 b_k1_DIS_s005 - 1.01 s_k1_DIS_s005  = 0
 w_k1_DIS_s006_return_constraint: 0.847698 w_k1_DIS_s00 - w_k1_DIS_s006
                                 + 0.99 b_k1_DIS_s006 - 1.01 s_k1_DIS_s006  = 0
 w_k1_DIS_s007_return_constraint: 1.73521 w_k1_DIS_s00 - w_k1_DIS_s007
                                 + 0.99 b_k1_DIS_s007 - 1.01 s_k1_DIS_s007  = 0
 w_k1_DIS_s010_return_constraint: 0.745815 w_k1_DIS_s01 - w_k1_DIS_s010
                                 + 0.99 b_k1_DIS_s010 - 1.01 s_k1_DIS_s010  = 0
 w_k1_DIS_s011_return_constraint: 1.10654 w_k1_DIS_s01 - w_k1_DIS_s011
                                 + 0.99 b_k1_DIS_s011 - 1.01 s_k1_DIS_s011  = 0
 w_k1_DIS_s012_return_constraint: 0.851461 w_k1_DIS_s01 - w_k1_DIS_s012
                                 + 0.99 b_k1_DIS_s012 - 1.01 s_k1_DIS_s012  = 0
 w_k1_DIS_s013_return_constraint: 1.10505 w_k1_DIS_s01 - w_k1_DIS_s013
                                 + 0.99 b_k1_DIS_s013 - 1.01 s_k1_DIS_s013  = 0
 w_k1_DIS_s014_return_constraint: 0.593974 w_k1_DIS_s01 - w_k1_DIS_s014
                                 + 0.99 b_k1_DIS_s014 - 1.01 s_k1_DIS_s014  = 0
 w_k1_DIS_s015_return_constraint: 1.21736 w_k1_DIS_s01 - w_k1_DIS_s015
                                 + 0.99 b_k1_DIS_s015 - 1.01 s_k1_DIS_s015  = 0
 w_k1_DIS_s016_return_constraint: 1.12881 w_k1_DIS_s01 - w_k1_DIS_s016
                                 + 0.99 b_k1_DIS_s016 - 1.01 s_k1_DIS_s016  = 0
 w_k1_DIS_s017_return_constraint: 1.59408 w_k1_DIS_s01 - w_k1_DIS_s017
                                 + 0.99 b_k1_DIS_s017 - 1.01 s_k1_DIS_s017  = 0
 w_k1_DIS_s100_return_constraint: 0.883938 w_k1_DIS_s10 - w_k1_DIS_s100
                                 + 0.99 b_k1_DIS_s100 - 1.01 s_k1_DIS_s100  = 0
 w_k1_DIS_s101_return_constraint: 1.74782 w_k1_DIS_s10 - w_k1_DIS_s101
                                 + 0.99 b_k1_DIS_s101 - 1.01 s_k1_DIS_s101  = 0
 w_k1_DIS_s102_return_constraint: 1.25188 w_k1_DIS_s10 - w_k1_DIS_s102
                                 + 0.99 b_k1_DIS_s102 - 1.01 s_k1_DIS_s102  = 0
 w_k1_DIS_s103_return_constraint: 1.53229 w_k1_DIS_s10 - w_k1_DIS_s103
                                 + 0.99 b_k1_DIS_s103 - 1.01 s_k1_DIS_s103  = 0
 w_k1_DIS_s104_return_constraint: 0.943223 w_k1_DIS_s10 - w_k1_DIS_s104
                                 + 0.99 b_k1_DIS_s104 - 1.01 s_k1_DIS_s104  = 0
 w_k1_DIS_s105_return_constraint: 0.642906 w_k1_DIS_s10 - w_k1_DIS_s105
                                 + 0.99 b_k1_DIS_s105 - 1.01 s_k1_DIS_s105  = 0
 w_k1_DIS_s106_return_constraint: 0.916532 w_k1_DIS_s10 - w_k1_DIS_s106
                                 + 0.99 b_k1_DIS_s106 - 1.01 s_k1_DIS_s106  = 0
 w_k1_DIS_s107_return_constraint: 0.823719 w_k1_DIS_s10 - w_k1_DIS_s107
                                 + 0.99 b_k1_DIS_s107 - 1.01 s_k1_DIS_s107  = 0
 w_k1_DIS_s110_return_constraint: 0.633012 w_k1_DIS_s11 - w_k1_DIS_s110
                                 + 0.99 b_k1_DIS_s110 - 1.01 s_k1_DIS_s110  = 0
 w_k1_DIS_s111_return_constraint: 1.43883 w_k1_DIS_s11 - w_k1_DIS_s111
                                 + 0.99 b_k1_DIS_s111 - 1.01 s_k1_DIS_s111  = 0
 w_k1_DIS_s112_return_constraint: 0.674699 w_k1_DIS_s11 - w_k1_DIS_s112
                                 + 0.99 b_k1_DIS_s112 - 1.01 s_k1_DIS_s112  = 0
 w_k1_DIS_s113_return_constraint: 0.842455 w_k1_DIS_s11 - w_k1_DIS_s113
                                 + 0.99 b_k1_DIS_s113 - 1.01 s_k1_DIS_s113  = 0
 w_k1_DIS_s114_return_constraint: 0.956235 w_k1_DIS_s11 - w_k1_DIS_s114
                                 + 0.99 b_k1_DIS_s114 - 1.01 s_k1_DIS_s114  = 0
 w_k1_DIS_s115_return_constraint: 1.31996 w_k1_DIS_s11 - w_k1_DIS_s115
                                 + 0.99 b_k1_DIS_s115 - 1.01 s_k1_DIS_s115  = 0
 w_k1_DIS_s116_return_constraint: 1.13518 w_k1_DIS_s11 - w_k1_DIS_s116
                                 + 0.99 b_k1_DIS_s116 - 1.01 s_k1_DIS_s116  = 0
 w_k1_DIS_s117_return_constraint: 0.931511 w_k1_DIS_s11 - w_k1_DIS_s117
                                 + 0.99 b_k1_DIS_s117 - 1.01 s_k1_DIS_s117  = 0
 w_k1_FDX_s0_return_constraint:  1.16455 w_0_FDX - w_k1_FDX_s0
                                 + 0.99 b_k1_FDX_s0 - 1.01 s_k1_FDX_s0  = 0
 w_k1_FDX_s1_return_constraint:  0.913212 w_0_FDX - w_k1_FDX_s1
                                 + 0.99 b_k1_FDX_s1 - 1.01 s_k1_FDX_s1  = 0
 w_k1_FDX_s00_return_constraint: 1.13123 w_k1_FDX_s0 - w_k1_FDX_s00
                                 + 0.99 b_k1_FDX_s00 - 1.01 s_k1_FDX_s00  = 0
 w_k1_FDX_s01_return_constraint: 0.825844 w_k1_FDX_s0 - w_k1_FDX_s01
                                 + 0.99 b_k1_FDX_s01 - 1.01 s_k1_FDX_s01  = 0
 w_k1_FDX_s10_return_constraint: 1.14829 w_k1_FDX_s1 - w_k1_FDX_s10
                                 + 0.99 b_k1_FDX_s10 - 1.01 s_k1_FDX_s10  = 0
 w_k1_FDX_s11_return_constraint: 0.802656 w_k1_FDX_s1 - w_k1_FDX_s11
                                 + 0.99 b_k1_FDX_s11 - 1.01 s_k1_FDX_s11  = 0
 w_k1_FDX_s000_return_constraint: 1.09555 w_k1_FDX_s00 - w_k1_FDX_s000
                                 + 0.99 b_k1_FDX_s000 - 1.01 s_k1_FDX_s000  = 0
 w_k1_FDX_s001_return_constraint: 1.31431 w_k1_FDX_s00 - w_k1_FDX_s001
                                 + 0.99 b_k1_FDX_s001 - 1.01 s_k1_FDX_s001  = 0
 w_k1_FDX_s002_return_constraint: 0.775898 w_k1_FDX_s00 - w_k1_FDX_s002
                                 + 0.99 b_k1_FDX_s002 - 1.01 s_k1_FDX_s002  = 0
 w_k1_FDX_s003_return_constraint: 1.10866 w_k1_FDX_s00 - w_k1_FDX_s003
                                 + 0.99 b_k1_FDX_s003 - 1.01 s_k1_FDX_s003  = 0
 w_k1_FDX_s004_return_constraint: 0.979559 w_k1_FDX_s00 - w_k1_FDX_s004
                                 + 0.99 b_k1_FDX_s004 - 1.01 s_k1_FDX_s004  = 0
 w_k1_FDX_s005_return_constraint: 0.973091 w_k1_FDX_s00 - w_k1_FDX_s005
                                 + 0.99 b_k1_FDX_s005 - 1.01 s_k1_FDX_s005  = 0
 w_k1_FDX_s006_return_constraint: 0.918976 w_k1_FDX_s00 - w_k1_FDX_s006
                                 + 0.99 b_k1_FDX_s006 - 1.01 s_k1_FDX_s006  = 0
 w_k1_FDX_s007_return_constraint: 1.57056 w_k1_FDX_s00 - w_k1_FDX_s007
                                 + 0.99 b_k1_FDX_s007 - 1.01 s_k1_FDX_s007  = 0
 w_k1_FDX_s010_return_constraint: 0.796196 w_k1_FDX_s01 - w_k1_FDX_s010
                                 + 0.99 b_k1_FDX_s010 - 1.01 s_k1_FDX_s010  = 0
 w_k1_FDX_s011_return_constraint: 0.983312 w_k1_FDX_s01 - w_k1_FDX_s011
                                 + 0.99 b_k1_FDX_s011 - 1.01 s_k1_FDX_s011  = 0
 w_k1_FDX_s012_return_constraint: 1.0406 w_k1_FDX_s01 - w_k1_FDX_s012
                                 + 0.99 b_k1_FDX_s012 - 1.01 s_k1_FDX_s012  = 0
 w_k1_FDX_s013_return_constraint: 0.919917 w_k1_FDX_s01 - w_k1_FDX_s013
                                 + 0.99 b_k1_FDX_s013 - 1.01 s_k1_FDX_s013  = 0
 w_k1_FDX_s014_return_constraint: 0.786474 w_k1_FDX_s01 - w_k1_FDX_s014
                                 + 0.99 b_k1_FDX_s014 - 1.01 s_k1_FDX_s014  = 0
 w_k1_FDX_s015_return_constraint: 1.25839 w_k1_FDX_s01 - w_k1_FDX_s015
                                 + 0.99 b_k1_FDX_s015 - 1.01 s_k1_FDX_s015  = 0
 w_k1_FDX_s016_return_constraint: 1.1237 w_k1_FDX_s01 - w_k1_FDX_s016
                                 + 0.99 b_k1_FDX_s016 - 1.01 s_k1_FDX_s016  = 0
 w_k1_FDX_s017_return_constraint: 1.38564 w_k1_FDX_s01 - w_k1_FDX_s017
                                 + 0.99 b_k1_FDX_s017 - 1.01 s_k1_FDX_s017  = 0
 w_k1_FDX_s100_return_constraint: 0.866199 w_k1_FDX_s10 - w_k1_FDX_s100
                                 + 0.99 b_k1_FDX_s100 - 1.01 s_k1_FDX_s100  = 0
 w_k1_FDX_s101_return_constraint: 1.19947 w_k1_FDX_s10 - w_k1_FDX_s101
                                 + 0.99 b_k1_FDX_s101 - 1.01 s_k1_FDX_s101  = 0
 w_k1_FDX_s102_return_constraint: 1.18879 w_k1_FDX_s10 - w_k1_FDX_s102
                                 + 0.99 b_k1_FDX_s102 - 1.01 s_k1_FDX_s102  = 0
 w_k1_FDX_s103_return_constraint: 1.28706 w_k1_FDX_s10 - w_k1_FDX_s103
                                 + 0.99 b_k1_FDX_s103 - 1.01 s_k1_FDX_s103  = 0
 w_k1_FDX_s104_return_constraint: 0.964054 w_k1_FDX_s10 - w_k1_FDX_s104
                                 + 0.99 b_k1_FDX_s104 - 1.01 s_k1_FDX_s104  = 0
 w_k1_FDX_s105_return_constraint: 0.741624 w_k1_FDX_s10 - w_k1_FDX_s105
                                 + 0.99 b_k1_FDX_s105 - 1.01 s_k1_FDX_s105  = 0
 w_k1_FDX_s106_return_constraint: 0.83756 w_k1_FDX_s10 - w_k1_FDX_s106
                                 + 0.99 b_k1_FDX_s106 - 1.01 s_k1_FDX_s106  = 0
 w_k1_FDX_s107_return_constraint: 0.962729 w_k1_FDX_s10 - w_k1_FDX_s107
                                 + 0.99 b_k1_FDX_s107 - 1.01 s_k1_FDX_s107  = 0
 w_k1_FDX_s110_return_constraint: 0.716204 w_k1_FDX_s11 - w_k1_FDX_s110
                                 + 0.99 b_k1_FDX_s110 - 1.01 s_k1_FDX_s110  = 0
 w_k1_FDX_s111_return_constraint: 1.25452 w_k1_FDX_s11 - w_k1_FDX_s111
                                 + 0.99 b_k1_FDX_s111 - 1.01 s_k1_FDX_s111  = 0
 w_k1_FDX_s112_return_constraint: 0.840608 w_k1_FDX_s11 - w_k1_FDX_s112
                                 + 0.99 b_k1_FDX_s112 - 1.01 s_k1_FDX_s112  = 0
 w_k1_FDX_s113_return_constraint: 0.954106 w_k1_FDX_s11 - w_k1_FDX_s113
                                 + 0.99 b_k1_FDX_s113 - 1.01 s_k1_FDX_s113  = 0
 w_k1_FDX_s114_return_constraint: 1.09466 w_k1_FDX_s11 - w_k1_FDX_s114
                                 + 0.99 b_k1_FDX_s114 - 1.01 s_k1_FDX_s114  = 0
 w_k1_FDX_s115_return_constraint: 1.27917 w_k1_FDX_s11 - w_k1_FDX_s115
                                 + 0.99 b_k1_FDX_s115 - 1.01 s_k1_FDX_s115  = 0
 w_k1_FDX_s116_return_constraint: 0.923254 w_k1_FDX_s11 - w_k1_FDX_s116
                                 + 0.99 b_k1_FDX_s116 - 1.01 s_k1_FDX_s116  = 0
 w_k1_FDX_s117_return_constraint: 0.935536 w_k1_FDX_s11 - w_k1_FDX_s117
                                 + 0.99 b_k1_FDX_s117 - 1.01 s_k1_FDX_s117  = 0
 w_k1_IBM_s0_return_constraint:  0.989729 w_0_IBM - w_k1_IBM_s0
                                 + 0.99 b_k1_IBM_s0 - 1.01 s_k1_IBM_s0  = 0
 w_k1_IBM_s1_return_constraint:  1.07064 w_0_IBM - w_k1_IBM_s1
                                 + 0.99 b_k1_IBM_s1 - 1.01 s_k1_IBM_s1  = 0
 w_k1_IBM_s00_return_constraint: 1.18567 w_k1_IBM_s0 - w_k1_IBM_s00
                                 + 0.99 b_k1_IBM_s00 - 1.01 s_k1_IBM_s00  = 0
 w_k1_IBM_s01_return_constraint: 0.748266 w_k1_IBM_s0 - w_k1_IBM_s01
                                 + 0.99 b_k1_IBM_s01 - 1.01 s_k1_IBM_s01  = 0
 w_k1_IBM_s10_return_constraint: 1.14828 w_k1_IBM_s1 - w_k1_IBM_s10
                                 + 0.99 b_k1_IBM_s10 - 1.01 s_k1_IBM_s10  = 0
 w_k1_IBM_s11_return_constraint: 0.756636 w_k1_IBM_s1 - w_k1_IBM_s11
                                 + 0.99 b_k1_IBM_s11 - 1.01 s_k1_IBM_s11  = 0
 w_k1_IBM_s000_return_constraint: 0.747944 w_k1_IBM_s00 - w_k1_IBM_s000
                                 + 0.99 b_k1_IBM_s000 - 1.01 s_k1_IBM_s000  = 0
 w_k1_IBM_s001_return_constraint: 0.831879 w_k1_IBM_s00 - w_k1_IBM_s001
                                 + 0.99 b_k1_IBM_s001 - 1.01 s_k1_IBM_s001  = 0
 w_k1_IBM_s002_return_constraint: 1.03097 w_k1_IBM_s00 - w_k1_IBM_s002
                                 + 0.99 b_k1_IBM_s002 - 1.01 s_k1_IBM_s002  = 0
 w_k1_IBM_s003_return_constraint: 1.49856 w_k1_IBM_s00 - w_k1_IBM_s003
                                 + 0.99 b_k1_IBM_s003 - 1.01 s_k1_IBM_s003  = 0
 w_k1_IBM_s004_return_constraint: 0.949343 w_k1_IBM_s00 - w_k1_IBM_s004
                                 + 0.99 b_k1_IBM_s004 - 1.01 s_k1_IBM_s004  = 0
 w_k1_IBM_s005_return_constraint: 0.512009 w_k1_IBM_s00 - w_k1_IBM_s005
                                 + 0.99 b_k1_IBM_s005 - 1.01 s_k1_IBM_s005  = 0
 w_k1_IBM_s006_return_constraint: 0.792206 w_k1_IBM_s00 - w_k1_IBM_s006
                                 + 0.99 b_k1_IBM_s006 - 1.01 s_k1_IBM_s006  = 0
 w_k1_IBM_s007_return_constraint: 1.22348 w_k1_IBM_s00 - w_k1_IBM_s007
                                 + 0.99 b_k1_IBM_s007 - 1.01 s_k1_IBM_s007  = 0
 w_k1_IBM_s010_return_constraint: 0.974535 w_k1_IBM_s01 - w_k1_IBM_s010
                                 + 0.99 b_k1_IBM_s010 - 1.01 s_k1_IBM_s010  = 0
 w_k1_IBM_s011_return_constraint: 1.12713 w_k1_IBM_s01 - w_k1_IBM_s011
                                 + 0.99 b_k1_IBM_s011 - 1.01 s_k1_IBM_s011  = 0
 w_k1_IBM_s012_return_constraint: 0.783984 w_k1_IBM_s01 - w_k1_IBM_s012
                                 + 0.99 b_k1_IBM_s012 - 1.01 s_k1_IBM_s012  = 0
 w_k1_IBM_s013_return_constraint: 1.7367 w_k1_IBM_s01 - w_k1_IBM_s013
                                 + 0.99 b_k1_IBM_s013 - 1.01 s_k1_IBM_s013  = 0
 w_k1_IBM_s014_return_constraint: 0.596723 w_k1_IBM_s01 - w_k1_IBM_s014
                                 + 0.99 b_k1_IBM_s014 - 1.01 s_k1_IBM_s014  = 0
 w_k1_IBM_s015_return_constraint: 0.8668 w_k1_IBM_s01 - w_k1_IBM_s015
                                 + 0.99 b_k1_IBM_s015 - 1.01 s_k1_IBM_s015  = 0
 w_k1_IBM_s016_return_constraint: 1.64563 w_k1_IBM_s01 - w_k1_IBM_s016
                                 + 0.99 b_k1_IBM_s016 - 1.01 s_k1_IBM_s016  = 0
 w_k1_IBM_s017_return_constraint: 1.33481 w_k1_IBM_s01 - w_k1_IBM_s017
                                 + 0.99 b_k1_IBM_s017 - 1.01 s_k1_IBM_s017  = 0
 w_k1_IBM_s100_return_constraint: 1.52434 w_k1_IBM_s10 - w_k1_IBM_s100
                                 + 0.99 b_k1_IBM_s100 - 1.01 s_k1_IBM_s100  = 0
 w_k1_IBM_s101_return_constraint: 2.00594 w_k1_IBM_s10 - w_k1_IBM_s101
                                 + 0.99 b_k1_IBM_s101 - 1.01 s_k1_IBM_s101  = 0
 w_k1_IBM_s102_return_constraint: 0.835538 w_k1_IBM_s10 - w_k1_IBM_s102
                                 + 0.99 b_k1_IBM_s102 - 1.01 s_k1_IBM_s102  = 0
 w_k1_IBM_s103_return_constraint: 1.28395 w_k1_IBM_s10 - w_k1_IBM_s103
                                 + 0.99 b_k1_IBM_s103 - 1.01 s_k1_IBM_s103  = 0
 w_k1_IBM_s104_return_constraint: 1.03609 w_k1_IBM_s10 - w_k1_IBM_s104
                                 + 0.99 b_k1_IBM_s104 - 1.01 s_k1_IBM_s104  = 0
 w_k1_IBM_s105_return_constraint: 0.809349 w_k1_IBM_s10 - w_k1_IBM_s105
                                 + 0.99 b_k1_IBM_s105 - 1.01 s_k1_IBM_s105  = 0
 w_k1_IBM_s106_return_constraint: 1.27142 w_k1_IBM_s10 - w_k1_IBM_s106
                                 + 0.99 b_k1_IBM_s106 - 1.01 s_k1_IBM_s106  = 0
 w_k1_IBM_s107_return_constraint: 0.584619 w_k1_IBM_s10 - w_k1_IBM_s107
                                 + 0.99 b_k1_IBM_s107 - 1.01 s_k1_IBM_s107  = 0
 w_k1_IBM_s110_return_constraint: 0.807096 w_k1_IBM_s11 - w_k1_IBM_s110
                                 + 0.99 b_k1_IBM_s110 - 1.01 s_k1_IBM_s110  = 0
 w_k1_IBM_s111_return_constraint: 1.52935 w_k1_IBM_s11 - w_k1_IBM_s111
                                 + 0.99 b_k1_IBM_s111 - 1.01 s_k1_IBM_s111  = 0
 w_k1_IBM_s112_return_constraint: 0.572238 w_k1_IBM_s11 - w_k1_IBM_s112
                                 + 0.99 b_k1_IBM_s112 - 1.01 s_k1_IBM_s112  = 0
 w_k1_IBM_s113_return_constraint: 0.884505 w_k1_IBM_s11 - w_k1_IBM_s113
                                 + 0.99 b_k1_IBM_s113 - 1.01 s_k1_IBM_s113  = 0
 w_k1_IBM_s114_return_constraint: 0.637417 w_k1_IBM_s11 - w_k1_IBM_s114
                                 + 0.99 b_k1_IBM_s114 - 1.01 s_k1_IBM_s114  = 0
 w_k1_IBM_s115_return_constraint: 1.02875 w_k1_IBM_s11 - w_k1_IBM_s115
                                 + 0.99 b_k1_IBM_s115 - 1.01 s_k1_IBM_s115  = 0
 w_k1_IBM_s116_return_constraint: 1.86569 w_k1_IBM_s11 - w_k1_IBM_s116
                                 + 0.99 b_k1_IBM_s116 - 1.01 s_k1_IBM_s116  = 0
 w_k1_IBM_s117_return_constraint: 1.11554 w_k1_IBM_s11 - w_k1_IBM_s117
                                 + 0.99 b_k1_IBM_s117 - 1.01 s_k1_IBM_s117  = 0
 w_k1_INTC_s0_return_constraint: 1.1632 w_0_INTC - w_k1_INTC_s0
                                 + 0.99 b_k1_INTC_s0 - 1.01 s_k1_INTC_s0  = 0
 w_k1_INTC_s1_return_constraint: 0.978617 w_0_INTC - w_k1_INTC_s1
                                 + 0.99 b_k1_INTC_s1 - 1.01 s_k1_INTC_s1  = 0
 w_k1_INTC_s00_return_constraint: 1.06868 w_k1_INTC_s0 - w_k1_INTC_s00
                                 + 0.99 b_k1_INTC_s00 - 1.01 s_k1_INTC_s00  = 0
 w_k1_INTC_s01_return_constraint: 0.888477 w_k1_INTC_s0 - w_k1_INTC_s01
                                 + 0.99 b_k1_INTC_s01 - 1.01 s_k1_INTC_s01  = 0
 w_k1_INTC_s10_return_constraint: 1.0258 w_k1_INTC_s1 - w_k1_INTC_s10
                                 + 0.99 b_k1_INTC_s10 - 1.01 s_k1_INTC_s10  = 0
 w_k1_INTC_s11_return_constraint: 0.843667 w_k1_INTC_s1 - w_k1_INTC_s11
                                 + 0.99 b_k1_INTC_s11 - 1.01 s_k1_INTC_s11  = 0
 w_k1_INTC_s000_return_constraint: 0.951644 w_k1_INTC_s00 - w_k1_INTC_s000
                                 + 0.99 b_k1_INTC_s000 - 1.01 s_k1_INTC_s000
                                  = 0
 w_k1_INTC_s001_return_constraint: 1.16462 w_k1_INTC_s00 - w_k1_INTC_s001
                                 + 0.99 b_k1_INTC_s001 - 1.01 s_k1_INTC_s001
                                  = 0
 w_k1_INTC_s002_return_constraint: 0.685205 w_k1_INTC_s00 - w_k1_INTC_s002
                                 + 0.99 b_k1_INTC_s002 - 1.01 s_k1_INTC_s002
                                  = 0
 w_k1_INTC_s003_return_constraint: 1.17102 w_k1_INTC_s00 - w_k1_INTC_s003
                                 + 0.99 b_k1_INTC_s003 - 1.01 s_k1_INTC_s003
                                  = 0
 w_k1_INTC_s004_return_constraint: 0.955596 w_k1_INTC_s00 - w_k1_INTC_s004
                                 + 0.99 b_k1_INTC_s004 - 1.01 s_k1_INTC_s004
                                  = 0
 w_k1_INTC_s005_return_constraint: 0.887154 w_k1_INTC_s00 - w_k1_INTC_s005
                                 + 0.99 b_k1_INTC_s005 - 1.01 s_k1_INTC_s005
                                  = 0
 w_k1_INTC_s006_return_constraint: 0.877142 w_k1_INTC_s00 - w_k1_INTC_s006
                                 + 0.99 b_k1_INTC_s006 - 1.01 s_k1_INTC_s006
                                  = 0
 w_k1_INTC_s007_return_constraint: 1.35085 w_k1_INTC_s00 - w_k1_INTC_s007
                                 + 0.99 b_k1_INTC_s007 - 1.01 s_k1_INTC_s007
                                  = 0
 w_k1_INTC_s010_return_constraint: 0.734906 w_k1_INTC_s01 - w_k1_INTC_s010
                                 + 0.99 b_k1_INTC_s010 - 1.01 s_k1_INTC_s010
                                  = 0
 w_k1_INTC_s011_return_constraint: 0.940637 w_k1_INTC_s01 - w_k1_INTC_s011
                                 + 0.99 b_k1_INTC_s011 - 1.01 s_k1_INTC_s011
                                  = 0
 w_k1_INTC_s012_return_constraint: 0.860635 w_k1_INTC_s01 - w_k1_INTC_s012
                                 + 0.99 b_k1_INTC_s012 - 1.01 s_k1_INTC_s012
                                  = 0
 w_k1_INTC_s013_return_constraint: 1.03462 w_k1_INTC_s01 - w_k1_INTC_s013
                                 + 0.99 b_k1_INTC_s013 - 1.01 s_k1_INTC_s013
                                  = 0
 w_k1_INTC_s014_return_constraint: 0.568699 w_k1_INTC_s01 - w_k1_INTC_s014
                                 + 0.99 b_k1_INTC_s014 - 1.01 s_k1_INTC_s014
                                  = 0
 w_k1_INTC_s015_return_constraint: 1.05051 w_k1_INTC_s01 - w_k1_INTC_s015
                                 + 0.99 b_k1_INTC_s015 - 1.01 s_k1_INTC_s015
                                  = 0
 w_k1_INTC_s016_return_constraint: 1.18585 w_k1_INTC_s01 - w_k1_INTC_s016
                                 + 0.99 b_k1_INTC_s016 - 1.01 s_k1_INTC_s016
                                  = 0
 w_k1_INTC_s017_return_constraint: 1.12066 w_k1_INTC_s01 - w_k1_INTC_s017
                                 + 0.99 b_k1_INTC_s017 - 1.01 s_k1_INTC_s017
                                  = 0
 w_k1_INTC_s100_return_constraint: 0.99058 w_k1_INTC_s10 - w_k1_INTC_s100
                                 + 0.99 b_k1_INTC_s100 - 1.01 s_k1_INTC_s100
                                  = 0
 w_k1_INTC_s101_return_constraint: 1.24196 w_k1_INTC_s10 - w_k1_INTC_s101
                                 + 0.99 b_k1_INTC_s101 - 1.01 s_k1_INTC_s101
                                  = 0
 w_k1_INTC_s102_return_constraint: 1.16614 w_k1_INTC_s10 - w_k1_INTC_s102
                                 + 0.99 b_k1_INTC_s102 - 1.01 s_k1_INTC_s102
                                  = 0
 w_k1_INTC_s103_return_constraint: 1.18074 w_k1_INTC_s10 - w_k1_INTC_s103
                                 + 0.99 b_k1_INTC_s103 - 1.01 s_k1_INTC_s103
                                  = 0
 w_k1_INTC_s104_return_constraint: 1.10182 w_k1_INTC_s10 - w_k1_INTC_s104
                                 + 0.99 b_k1_INTC_s104 - 1.01 s_k1_INTC_s104
                                  = 0
 w_k1_INTC_s105_return_constraint: 0.816472 w_k1_INTC_s10 - w_k1_INTC_s105
                                 + 0.99 b_k1_INTC_s105 - 1.01 s_k1_INTC_s105
                                  = 0
 w_k1_INTC_s106_return_constraint: 0.977751 w_k1_INTC_s10 - w_k1_INTC_s106
                                 + 0.99 b_k1_INTC_s106 - 1.01 s_k1_INTC_s106
                                  = 0
 w_k1_INTC_s107_return_constraint: 0.910679 w_k1_INTC_s10 - w_k1_INTC_s107
                                 + 0.99 b_k1_INTC_s107 - 1.01 s_k1_INTC_s107
                                  = 0
 w_k1_INTC_s110_return_constraint: 0.802516 w_k1_INTC_s11 - w_k1_INTC_s110
                                 + 0.99 b_k1_INTC_s110 - 1.01 s_k1_INTC_s110
                                  = 0
 w_k1_INTC_s111_return_constraint: 1.25208 w_k1_INTC_s11 - w_k1_INTC_s111
                                 + 0.99 b_k1_INTC_s111 - 1.01 s_k1_INTC_s111
                                  = 0
 w_k1_INTC_s112_return_constraint: 0.81284 w_k1_INTC_s11 - w_k1_INTC_s112
                                 + 0.99 b_k1_INTC_s112 - 1.01 s_k1_INTC_s112
                                  = 0
 w_k1_INTC_s113_return_constraint: 0.898248 w_k1_INTC_s11 - w_k1_INTC_s113
                                 + 0.99 b_k1_INTC_s113 - 1.01 s_k1_INTC_s113
                                  = 0
 w_k1_INTC_s114_return_constraint: 0.934427 w_k1_INTC_s11 - w_k1_INTC_s114
                                 + 0.99 b_k1_INTC_s114 - 1.01 s_k1_INTC_s114
                                  = 0
 w_k1_INTC_s115_return_constraint: 1.28717 w_k1_INTC_s11 - w_k1_INTC_s115
                                 + 0.99 b_k1_INTC_s115 - 1.01 s_k1_INTC_s115
                                  = 0
 w_k1_INTC_s116_return_constraint: 1.06993 w_k1_INTC_s11 - w_k1_INTC_s116
                                 + 0.99 b_k1_INTC_s116 - 1.01 s_k1_INTC_s116
                                  = 0
 w_k1_INTC_s117_return_constraint: 0.901965 w_k1_INTC_s11 - w_k1_INTC_s117
                                 + 0.99 b_k1_INTC_s117 - 1.01 s_k1_INTC_s117
                                  = 0
 w_k1_KO_s0_return_constraint:   1.06813 w_0_KO - w_k1_KO_s0 + 0.99 b_k1_KO_s0
                                 - 1.01 s_k1_KO_s0  = 0
 w_k1_KO_s1_return_constraint:   0.951705 w_0_KO - w_k1_KO_s1 + 0.99 b_k1_KO_s1
                                 - 1.01 s_k1_KO_s1  = 0
 w_k1_KO_s00_return_constraint:  1.18161 w_k1_KO_s0 - w_k1_KO_s00
                                 + 0.99 b_k1_KO_s00 - 1.01 s_k1_KO_s00  = 0
 w_k1_KO_s01_return_constraint:  0.83609 w_k1_KO_s0 - w_k1_KO_s01
                                 + 0.99 b_k1_KO_s01 - 1.01 s_k1_KO_s01  = 0
 w_k1_KO_s10_return_constraint:  1.25431 w_k1_KO_s1 - w_k1_KO_s10
                                 + 0.99 b_k1_KO_s10 - 1.01 s_k1_KO_s10  = 0
 w_k1_KO_s11_return_constraint:  0.753489 w_k1_KO_s1 - w_k1_KO_s11
                                 + 0.99 b_k1_KO_s11 - 1.01 s_k1_KO_s11  = 0
 w_k1_KO_s000_return_constraint: 0.974286 w_k1_KO_s00 - w_k1_KO_s000
                                 + 0.99 b_k1_KO_s000 - 1.01 s_k1_KO_s000  = 0
 w_k1_KO_s001_return_constraint: 1.07788 w_k1_KO_s00 - w_k1_KO_s001
                                 + 0.99 b_k1_KO_s001 - 1.01 s_k1_KO_s001  = 0
 w_k1_KO_s002_return_constraint: 0.899361 w_k1_KO_s00 - w_k1_KO_s002
                                 + 0.99 b_k1_KO_s002 - 1.01 s_k1_KO_s002  = 0
 w_k1_KO_s003_return_constraint: 1.27413 w_k1_KO_s00 - w_k1_KO_s003
                                 + 0.99 b_k1_KO_s003 - 1.01 s_k1_KO_s003  = 0
 w_k1_KO_s004_return_constraint: 0.82824 w_k1_KO_s00 - w_k1_KO_s004
                                 + 0.99 b_k1_KO_s004 - 1.01 s_k1_KO_s004  = 0
 w_k1_KO_s005_return_constraint: 0.683521 w_k1_KO_s00 - w_k1_KO_s005
                                 + 0.99 b_k1_KO_s005 - 1.01 s_k1_KO_s005  = 0
 w_k1_KO_s006_return_constraint: 0.902056 w_k1_KO_s00 - w_k1_KO_s006
                                 + 0.99 b_k1_KO_s006 - 1.01 s_k1_KO_s006  = 0
 w_k1_KO_s007_return_constraint: 1.35909 w_k1_KO_s00 - w_k1_KO_s007
                                 + 0.99 b_k1_KO_s007 - 1.01 s_k1_KO_s007  = 0
 w_k1_KO_s010_return_constraint: 0.749251 w_k1_KO_s01 - w_k1_KO_s010
                                 + 0.99 b_k1_KO_s010 - 1.01 s_k1_KO_s010  = 0
 w_k1_KO_s011_return_constraint: 1.05336 w_k1_KO_s01 - w_k1_KO_s011
                                 + 0.99 b_k1_KO_s011 - 1.01 s_k1_KO_s011  = 0
 w_k1_KO_s012_return_constraint: 0.861084 w_k1_KO_s01 - w_k1_KO_s012
                                 + 0.99 b_k1_KO_s012 - 1.01 s_k1_KO_s012  = 0
 w_k1_KO_s013_return_constraint: 1.19268 w_k1_KO_s01 - w_k1_KO_s013
                                 + 0.99 b_k1_KO_s013 - 1.01 s_k1_KO_s013  = 0
 w_k1_KO_s014_return_constraint: 0.650399 w_k1_KO_s01 - w_k1_KO_s014
                                 + 0.99 b_k1_KO_s014 - 1.01 s_k1_KO_s014  = 0
 w_k1_KO_s015_return_constraint: 0.955939 w_k1_KO_s01 - w_k1_KO_s015
                                 + 0.99 b_k1_KO_s015 - 1.01 s_k1_KO_s015  = 0
 w_k1_KO_s016_return_constraint: 1.22233 w_k1_KO_s01 - w_k1_KO_s016
                                 + 0.99 b_k1_KO_s016 - 1.01 s_k1_KO_s016  = 0
 w_k1_KO_s017_return_constraint: 1.25876 w_k1_KO_s01 - w_k1_KO_s017
                                 + 0.99 b_k1_KO_s017 - 1.01 s_k1_KO_s017  = 0
 w_k1_KO_s100_return_constraint: 0.988779 w_k1_KO_s10 - w_k1_KO_s100
                                 + 0.99 b_k1_KO_s100 - 1.01 s_k1_KO_s100  = 0
 w_k1_KO_s101_return_constraint: 1.6381 w_k1_KO_s10 - w_k1_KO_s101
                                 + 0.99 b_k1_KO_s101 - 1.01 s_k1_KO_s101  = 0
 w_k1_KO_s102_return_constraint: 0.980717 w_k1_KO_s10 - w_k1_KO_s102
                                 + 0.99 b_k1_KO_s102 - 1.01 s_k1_KO_s102  = 0
 w_k1_KO_s103_return_constraint: 1.22754 w_k1_KO_s10 - w_k1_KO_s103
                                 + 0.99 b_k1_KO_s103 - 1.01 s_k1_KO_s103  = 0
 w_k1_KO_s104_return_constraint: 0.881231 w_k1_KO_s10 - w_k1_KO_s104
                                 + 0.99 b_k1_KO_s104 - 1.01 s_k1_KO_s104  = 0
 w_k1_KO_s105_return_constraint: 0.6906 w_k1_KO_s10 - w_k1_KO_s105
                                 + 0.99 b_k1_KO_s105 - 1.01 s_k1_KO_s105  = 0
 w_k1_KO_s106_return_constraint: 0.98419 w_k1_KO_s10 - w_k1_KO_s106
                                 + 0.99 b_k1_KO_s106 - 1.01 s_k1_KO_s106  = 0
 w_k1_KO_s107_return_constraint: 0.765926 w_k1_KO_s10 - w_k1_KO_s107
                                 + 0.99 b_k1_KO_s107 - 1.01 s_k1_KO_s107  = 0
 w_k1_KO_s110_return_constraint: 0.817527 w_k1_KO_s11 - w_k1_KO_s110
                                 + 0.99 b_k1_KO_s110 - 1.01 s_k1_KO_s110  = 0
 w_k1_KO_s111_return_constraint: 1.41429 w_k1_KO_s11 - w_k1_KO_s111
                                 + 0.99 b_k1_KO_s111 - 1.01 s_k1_KO_s111  = 0
 w_k1_KO_s112_return_constraint: 0.675619 w_k1_KO_s11 - w_k1_KO_s112
                                 + 0.99 b_k1_KO_s112 - 1.01 s_k1_KO_s112  = 0
 w_k1_KO_s113_return_constraint: 0.894851 w_k1_KO_s11 - w_k1_KO_s113
                                 + 0.99 b_k1_KO_s113 - 1.01 s_k1_KO_s113  = 0
 w_k1_KO_s114_return_constraint: 0.828495 w_k1_KO_s11 - w_k1_KO_s114
                                 + 0.99 b_k1_KO_s114 - 1.01 s_k1_KO_s114  = 0
 w_k1_KO_s115_return_constraint: 1.22876 w_k1_KO_s11 - w_k1_KO_s115
                                 + 0.99 b_k1_KO_s115 - 1.01 s_k1_KO_s115  = 0
 w_k1_KO_s116_return_constraint: 1.37938 w_k1_KO_s11 - w_k1_KO_s116
                                 + 0.99 b_k1_KO_s116 - 1.01 s_k1_KO_s116  = 0
 w_k1_KO_s117_return_constraint: 1.09119 w_k1_KO_s11 - w_k1_KO_s117
                                 + 0.99 b_k1_KO_s117 - 1.01 s_k1_KO_s117  = 0
 w_k1_MAT_s0_return_constraint:  0.986301 w_0_MAT - w_k1_MAT_s0
                                 + 0.99 b_k1_MAT_s0 - 1.01 s_k1_MAT_s0  = 0
 w_k1_MAT_s1_return_constraint:  1.00655 w_0_MAT - w_k1_MAT_s1
                                 + 0.99 b_k1_MAT_s1 - 1.01 s_k1_MAT_s1  = 0
 w_k1_MAT_s00_return_constraint: 1.29651 w_k1_MAT_s0 - w_k1_MAT_s00
                                 + 0.99 b_k1_MAT_s00 - 1.01 s_k1_MAT_s00  = 0
 w_k1_MAT_s01_return_constraint: 0.78585 w_k1_MAT_s0 - w_k1_MAT_s01
                                 + 0.99 b_k1_MAT_s01 - 1.01 s_k1_MAT_s01  = 0
 w_k1_MAT_s10_return_constraint: 1.30861 w_k1_MAT_s1 - w_k1_MAT_s10
                                 + 0.99 b_k1_MAT_s10 - 1.01 s_k1_MAT_s10  = 0
 w_k1_MAT_s11_return_constraint: 0.697213 w_k1_MAT_s1 - w_k1_MAT_s11
                                 + 0.99 b_k1_MAT_s11 - 1.01 s_k1_MAT_s11  = 0
 w_k1_MAT_s000_return_constraint: 0.756256 w_k1_MAT_s00 - w_k1_MAT_s000
                                 + 0.99 b_k1_MAT_s000 - 1.01 s_k1_MAT_s000  = 0
 w_k1_MAT_s001_return_constraint: 0.743491 w_k1_MAT_s00 - w_k1_MAT_s001
                                 + 0.99 b_k1_MAT_s001 - 1.01 s_k1_MAT_s001  = 0
 w_k1_MAT_s002_return_constraint: 1.11562 w_k1_MAT_s00 - w_k1_MAT_s002
                                 + 0.99 b_k1_MAT_s002 - 1.01 s_k1_MAT_s002  = 0
 w_k1_MAT_s003_return_constraint: 1.64686 w_k1_MAT_s00 - w_k1_MAT_s003
                                 + 0.99 b_k1_MAT_s003 - 1.01 s_k1_MAT_s003  = 0
 w_k1_MAT_s004_return_constraint: 1.13464 w_k1_MAT_s00 - w_k1_MAT_s004
                                 + 0.99 b_k1_MAT_s004 - 1.01 s_k1_MAT_s004  = 0
 w_k1_MAT_s005_return_constraint: 0.576927 w_k1_MAT_s00 - w_k1_MAT_s005
                                 + 0.99 b_k1_MAT_s005 - 1.01 s_k1_MAT_s005  = 0
 w_k1_MAT_s006_return_constraint: 0.603827 w_k1_MAT_s00 - w_k1_MAT_s006
                                 + 0.99 b_k1_MAT_s006 - 1.01 s_k1_MAT_s006  = 0
 w_k1_MAT_s007_return_constraint: 1.61678 w_k1_MAT_s00 - w_k1_MAT_s007
                                 + 0.99 b_k1_MAT_s007 - 1.01 s_k1_MAT_s007  = 0
 w_k1_MAT_s010_return_constraint: 0.838157 w_k1_MAT_s01 - w_k1_MAT_s010
                                 + 0.99 b_k1_MAT_s010 - 1.01 s_k1_MAT_s010  = 0
 w_k1_MAT_s011_return_constraint: 1.04589 w_k1_MAT_s01 - w_k1_MAT_s011
                                 + 0.99 b_k1_MAT_s011 - 1.01 s_k1_MAT_s011  = 0
 w_k1_MAT_s012_return_constraint: 0.775703 w_k1_MAT_s01 - w_k1_MAT_s012
                                 + 0.99 b_k1_MAT_s012 - 1.01 s_k1_MAT_s012  = 0
 w_k1_MAT_s013_return_constraint: 1.64911 w_k1_MAT_s01 - w_k1_MAT_s013
                                 + 0.99 b_k1_MAT_s013 - 1.01 s_k1_MAT_s013  = 0
 w_k1_MAT_s014_return_constraint: 0.539711 w_k1_MAT_s01 - w_k1_MAT_s014
                                 + 0.99 b_k1_MAT_s014 - 1.01 s_k1_MAT_s014  = 0
 w_k1_MAT_s015_return_constraint: 0.906954 w_k1_MAT_s01 - w_k1_MAT_s015
                                 + 0.99 b_k1_MAT_s015 - 1.01 s_k1_MAT_s015  = 0
 w_k1_MAT_s016_return_constraint: 1.51505 w_k1_MAT_s01 - w_k1_MAT_s016
                                 + 0.99 b_k1_MAT_s016 - 1.01 s_k1_MAT_s016  = 0
 w_k1_MAT_s017_return_constraint: 1.34065 w_k1_MAT_s01 - w_k1_MAT_s017
                                 + 0.99 b_k1_MAT_s017 - 1.01 s_k1_MAT_s017  = 0
 w_k1_MAT_s100_return_constraint: 1.38808 w_k1_MAT_s10 - w_k1_MAT_s100
                                 + 0.99 b_k1_MAT_s100 - 1.01 s_k1_MAT_s100  = 0
 w_k1_MAT_s101_return_constraint: 1.97544 w_k1_MAT_s10 - w_k1_MAT_s101
                                 + 0.99 b_k1_MAT_s101 - 1.01 s_k1_MAT_s101  = 0
 w_k1_MAT_s102_return_constraint: 1.00447 w_k1_MAT_s10 - w_k1_MAT_s102
                                 + 0.99 b_k1_MAT_s102 - 1.01 s_k1_MAT_s102  = 0
 w_k1_MAT_s103_return_constraint: 1.28431 w_k1_MAT_s10 - w_k1_MAT_s103
                                 + 0.99 b_k1_MAT_s103 - 1.01 s_k1_MAT_s103  = 0
 w_k1_MAT_s104_return_constraint: 0.992532 w_k1_MAT_s10 - w_k1_MAT_s104
                                 + 0.99 b_k1_MAT_s104 - 1.01 s_k1_MAT_s104  = 0
 w_k1_MAT_s105_return_constraint: 0.677609 w_k1_MAT_s10 - w_k1_MAT_s105
                                 + 0.99 b_k1_MAT_s105 - 1.01 s_k1_MAT_s105  = 0
 w_k1_MAT_s106_return_constraint: 1.20009 w_k1_MAT_s10 - w_k1_MAT_s106
                                 + 0.99 b_k1_MAT_s106 - 1.01 s_k1_MAT_s106  = 0
 w_k1_MAT_s107_return_constraint: 0.603987 w_k1_MAT_s10 - w_k1_MAT_s107
                                 + 0.99 b_k1_MAT_s107 - 1.01 s_k1_MAT_s107  = 0
 w_k1_MAT_s110_return_constraint: 0.792393 w_k1_MAT_s11 - w_k1_MAT_s110
                                 + 0.99 b_k1_MAT_s110 - 1.01 s_k1_MAT_s110  = 0
 w_k1_MAT_s111_return_constraint: 2.10408 w_k1_MAT_s11 - w_k1_MAT_s111
                                 + 0.99 b_k1_MAT_s111 - 1.01 s_k1_MAT_s111  = 0
 w_k1_MAT_s112_return_constraint: 0.516988 w_k1_MAT_s11 - w_k1_MAT_s112
                                 + 0.99 b_k1_MAT_s112 - 1.01 s_k1_MAT_s112  = 0
 w_k1_MAT_s113_return_constraint: 1.10106 w_k1_MAT_s11 - w_k1_MAT_s113
                                 + 0.99 b_k1_MAT_s113 - 1.01 s_k1_MAT_s113  = 0
 w_k1_MAT_s114_return_constraint: 0.753904 w_k1_MAT_s11 - w_k1_MAT_s114
                                 + 0.99 b_k1_MAT_s114 - 1.01 s_k1_MAT_s114  = 0
 w_k1_MAT_s115_return_constraint: 0.955194 w_k1_MAT_s11 - w_k1_MAT_s115
                                 + 0.99 b_k1_MAT_s115 - 1.01 s_k1_MAT_s115  = 0
 w_k1_MAT_s116_return_constraint: 1.76179 w_k1_MAT_s11 - w_k1_MAT_s116
                                 + 0.99 b_k1_MAT_s116 - 1.01 s_k1_MAT_s116  = 0
 w_k1_MAT_s117_return_constraint: 0.981751 w_k1_MAT_s11 - w_k1_MAT_s117
                                 + 0.99 b_k1_MAT_s117 - 1.01 s_k1_MAT_s117  = 0
 w_k1_MSFT_s0_return_constraint: 1.11106 w_0_MSFT - w_k1_MSFT_s0
                                 + 0.99 b_k1_MSFT_s0 - 1.01 s_k1_MSFT_s0  = 0
 w_k1_MSFT_s1_return_constraint: 0.900045 w_0_MSFT - w_k1_MSFT_s1
                                 + 0.99 b_k1_MSFT_s1 - 1.01 s_k1_MSFT_s1  = 0
 w_k1_MSFT_s00_return_constraint: 1.17575 w_k1_MSFT_s0 - w_k1_MSFT_s00
                                 + 0.99 b_k1_MSFT_s00 - 1.01 s_k1_MSFT_s00  = 0
 w_k1_MSFT_s01_return_constraint: 0.867733 w_k1_MSFT_s0 - w_k1_MSFT_s01
                                 + 0.99 b_k1_MSFT_s01 - 1.01 s_k1_MSFT_s01  = 0
 w_k1_MSFT_s10_return_constraint: 1.13302 w_k1_MSFT_s1 - w_k1_MSFT_s10
                                 + 0.99 b_k1_MSFT_s10 - 1.01 s_k1_MSFT_s10  = 0
 w_k1_MSFT_s11_return_constraint: 0.856806 w_k1_MSFT_s1 - w_k1_MSFT_s11
                                 + 0.99 b_k1_MSFT_s11 - 1.01 s_k1_MSFT_s11  = 0
 w_k1_MSFT_s000_return_constraint: 0.996712 w_k1_MSFT_s00 - w_k1_MSFT_s000
                                 + 0.99 b_k1_MSFT_s000 - 1.01 s_k1_MSFT_s000
                                  = 0
 w_k1_MSFT_s001_return_constraint: 1.08241 w_k1_MSFT_s00 - w_k1_MSFT_s001
                                 + 0.99 b_k1_MSFT_s001 - 1.01 s_k1_MSFT_s001
                                  = 0
 w_k1_MSFT_s002_return_constraint: 0.735422 w_k1_MSFT_s00 - w_k1_MSFT_s002
                                 + 0.99 b_k1_MSFT_s002 - 1.01 s_k1_MSFT_s002
                                  = 0
 w_k1_MSFT_s003_return_constraint: 1.2116 w_k1_MSFT_s00 - w_k1_MSFT_s003
                                 + 0.99 b_k1_MSFT_s003 - 1.01 s_k1_MSFT_s003
                                  = 0
 w_k1_MSFT_s004_return_constraint: 0.86276 w_k1_MSFT_s00 - w_k1_MSFT_s004
                                 + 0.99 b_k1_MSFT_s004 - 1.01 s_k1_MSFT_s004
                                  = 0
 w_k1_MSFT_s005_return_constraint: 0.875379 w_k1_MSFT_s00 - w_k1_MSFT_s005
                                 + 0.99 b_k1_MSFT_s005 - 1.01 s_k1_MSFT_s005
                                  = 0
 w_k1_MSFT_s006_return_constraint: 0.868648 w_k1_MSFT_s00 - w_k1_MSFT_s006
                                 + 0.99 b_k1_MSFT_s006 - 1.01 s_k1_MSFT_s006
                                  = 0
 w_k1_MSFT_s007_return_constraint: 1.35566 w_k1_MSFT_s00 - w_k1_MSFT_s007
                                 + 0.99 b_k1_MSFT_s007 - 1.01 s_k1_MSFT_s007
                                  = 0
 w_k1_MSFT_s010_return_constraint: 0.882799 w_k1_MSFT_s01 - w_k1_MSFT_s010
                                 + 0.99 b_k1_MSFT_s010 - 1.01 s_k1_MSFT_s010
                                  = 0
 w_k1_MSFT_s011_return_constraint: 1.03477 w_k1_MSFT_s01 - w_k1_MSFT_s011
                                 + 0.99 b_k1_MSFT_s011 - 1.01 s_k1_MSFT_s011
                                  = 0
 w_k1_MSFT_s012_return_constraint: 1.03776 w_k1_MSFT_s01 - w_k1_MSFT_s012
                                 + 0.99 b_k1_MSFT_s012 - 1.01 s_k1_MSFT_s012
                                  = 0
 w_k1_MSFT_s013_return_constraint: 0.967613 w_k1_MSFT_s01 - w_k1_MSFT_s013
                                 + 0.99 b_k1_MSFT_s013 - 1.01 s_k1_MSFT_s013
                                  = 0
 w_k1_MSFT_s014_return_constraint: 0.780988 w_k1_MSFT_s01 - w_k1_MSFT_s014
                                 + 0.99 b_k1_MSFT_s014 - 1.01 s_k1_MSFT_s014
                                  = 0
 w_k1_MSFT_s015_return_constraint: 1.19178 w_k1_MSFT_s01 - w_k1_MSFT_s015
                                 + 0.99 b_k1_MSFT_s015 - 1.01 s_k1_MSFT_s015
                                  = 0
 w_k1_MSFT_s016_return_constraint: 1.05724 w_k1_MSFT_s01 - w_k1_MSFT_s016
                                 + 0.99 b_k1_MSFT_s016 - 1.01 s_k1_MSFT_s016
                                  = 0
 w_k1_MSFT_s017_return_constraint: 1.19938 w_k1_MSFT_s01 - w_k1_MSFT_s017
                                 + 0.99 b_k1_MSFT_s017 - 1.01 s_k1_MSFT_s017
                                  = 0
 w_k1_MSFT_s100_return_constraint: 0.837985 w_k1_MSFT_s10 - w_k1_MSFT_s100
                                 + 0.99 b_k1_MSFT_s100 - 1.01 s_k1_MSFT_s100
                                  = 0
 w_k1_MSFT_s101_return_constraint: 1.2855 w_k1_MSFT_s10 - w_k1_MSFT_s101
                                 + 0.99 b_k1_MSFT_s101 - 1.01 s_k1_MSFT_s101
                                  = 0
 w_k1_MSFT_s102_return_constraint: 1.16605 w_k1_MSFT_s10 - w_k1_MSFT_s102
                                 + 0.99 b_k1_MSFT_s102 - 1.01 s_k1_MSFT_s102
                                  = 0
 w_k1_MSFT_s103_return_constraint: 1.26384 w_k1_MSFT_s10 - w_k1_MSFT_s103
                                 + 0.99 b_k1_MSFT_s103 - 1.01 s_k1_MSFT_s103
                                  = 0
 w_k1_MSFT_s104_return_constraint: 1.01746 w_k1_MSFT_s10 - w_k1_MSFT_s104
                                 + 0.99 b_k1_MSFT_s104 - 1.01 s_k1_MSFT_s104
                                  = 0
 w_k1_MSFT_s105_return_constraint: 0.811703 w_k1_MSFT_s10 - w_k1_MSFT_s105
                                 + 0.99 b_k1_MSFT_s105 - 1.01 s_k1_MSFT_s105
                                  = 0
 w_k1_MSFT_s106_return_constraint: 0.924647 w_k1_MSFT_s10 - w_k1_MSFT_s106
                                 + 0.99 b_k1_MSFT_s106 - 1.01 s_k1_MSFT_s106
                                  = 0
 w_k1_MSFT_s107_return_constraint: 0.949297 w_k1_MSFT_s10 - w_k1_MSFT_s107
                                 + 0.99 b_k1_MSFT_s107 - 1.01 s_k1_MSFT_s107
                                  = 0
 w_k1_MSFT_s110_return_constraint: 0.779487 w_k1_MSFT_s11 - w_k1_MSFT_s110
                                 + 0.99 b_k1_MSFT_s110 - 1.01 s_k1_MSFT_s110
                                  = 0
 w_k1_MSFT_s111_return_constraint: 1.25425 w_k1_MSFT_s11 - w_k1_MSFT_s111
                                 + 0.99 b_k1_MSFT_s111 - 1.01 s_k1_MSFT_s111
                                  = 0
 w_k1_MSFT_s112_return_constraint: 0.820716 w_k1_MSFT_s11 - w_k1_MSFT_s112
                                 + 0.99 b_k1_MSFT_s112 - 1.01 s_k1_MSFT_s112
                                  = 0
 w_k1_MSFT_s113_return_constraint: 0.936568 w_k1_MSFT_s11 - w_k1_MSFT_s113
                                 + 0.99 b_k1_MSFT_s113 - 1.01 s_k1_MSFT_s113
                                  = 0
 w_k1_MSFT_s114_return_constraint: 0.990046 w_k1_MSFT_s11 - w_k1_MSFT_s114
                                 + 0.99 b_k1_MSFT_s114 - 1.01 s_k1_MSFT_s114
                                  = 0
 w_k1_MSFT_s115_return_constraint: 1.24006 w_k1_MSFT_s11 - w_k1_MSFT_s115
                                 + 0.99 b_k1_MSFT_s115 - 1.01 s_k1_MSFT_s115
                                  = 0
 w_k1_MSFT_s116_return_constraint: 0.974262 w_k1_MSFT_s11 - w_k1_MSFT_s116
                                 + 0.99 b_k1_MSFT_s116 - 1.01 s_k1_MSFT_s116
                                  = 0
 w_k1_MSFT_s117_return_constraint: 0.872413 w_k1_MSFT_s11 - w_k1_MSFT_s117
                                 + 0.99 b_k1_MSFT_s117 - 1.01 s_k1_MSFT_s117
                                  = 0
 w_k1_PG_s0_return_constraint:   1.06885 w_0_PG - w_k1_PG_s0 + 0.99 b_k1_PG_s0
                                 - 1.01 s_k1_PG_s0  = 0
 w_k1_PG_s1_return_constraint:   0.945415 w_0_PG - w_k1_PG_s1 + 0.99 b_k1_PG_s1
                                 - 1.01 s_k1_PG_s1  = 0
 w_k1_PG_s00_return_constraint:  1.12812 w_k1_PG_s0 - w_k1_PG_s00
                                 + 0.99 b_k1_PG_s00 - 1.01 s_k1_PG_s00  = 0
 w_k1_PG_s01_return_constraint:  0.842947 w_k1_PG_s0 - w_k1_PG_s01
                                 + 0.99 b_k1_PG_s01 - 1.01 s_k1_PG_s01  = 0
 w_k1_PG_s10_return_constraint:  1.14897 w_k1_PG_s1 - w_k1_PG_s10
                                 + 0.99 b_k1_PG_s10 - 1.01 s_k1_PG_s10  = 0
 w_k1_PG_s11_return_constraint:  0.867654 w_k1_PG_s1 - w_k1_PG_s11
                                 + 0.99 b_k1_PG_s11 - 1.01 s_k1_PG_s11  = 0
 w_k1_PG_s000_return_constraint: 0.910342 w_k1_PG_s00 - w_k1_PG_s000
                                 + 0.99 b_k1_PG_s000 - 1.01 s_k1_PG_s000  = 0
 w_k1_PG_s001_return_constraint: 1.04314 w_k1_PG_s00 - w_k1_PG_s001
                                 + 0.99 b_k1_PG_s001 - 1.01 s_k1_PG_s001  = 0
 w_k1_PG_s002_return_constraint: 0.88136 w_k1_PG_s00 - w_k1_PG_s002
                                 + 0.99 b_k1_PG_s002 - 1.01 s_k1_PG_s002  = 0
 w_k1_PG_s003_return_constraint: 1.11067 w_k1_PG_s00 - w_k1_PG_s003
                                 + 0.99 b_k1_PG_s003 - 1.01 s_k1_PG_s003  = 0
 w_k1_PG_s004_return_constraint: 0.928619 w_k1_PG_s00 - w_k1_PG_s004
                                 + 0.99 b_k1_PG_s004 - 1.01 s_k1_PG_s004  = 0
 w_k1_PG_s005_return_constraint: 0.93062 w_k1_PG_s00 - w_k1_PG_s005
                                 + 0.99 b_k1_PG_s005 - 1.01 s_k1_PG_s005  = 0
 w_k1_PG_s006_return_constraint: 0.885863 w_k1_PG_s00 - w_k1_PG_s006
                                 + 0.99 b_k1_PG_s006 - 1.01 s_k1_PG_s006  = 0
 w_k1_PG_s007_return_constraint: 1.35864 w_k1_PG_s00 - w_k1_PG_s007
                                 + 0.99 b_k1_PG_s007 - 1.01 s_k1_PG_s007  = 0
 w_k1_PG_s010_return_constraint: 0.885152 w_k1_PG_s01 - w_k1_PG_s010
                                 + 0.99 b_k1_PG_s010 - 1.01 s_k1_PG_s010  = 0
 w_k1_PG_s011_return_constraint: 1.03317 w_k1_PG_s01 - w_k1_PG_s011
                                 + 0.99 b_k1_PG_s011 - 1.01 s_k1_PG_s011  = 0
 w_k1_PG_s012_return_constraint: 0.929353 w_k1_PG_s01 - w_k1_PG_s012
                                 + 0.99 b_k1_PG_s012 - 1.01 s_k1_PG_s012  = 0
 w_k1_PG_s013_return_constraint: 1.14023 w_k1_PG_s01 - w_k1_PG_s013
                                 + 0.99 b_k1_PG_s013 - 1.01 s_k1_PG_s013  = 0
 w_k1_PG_s014_return_constraint: 0.808714 w_k1_PG_s01 - w_k1_PG_s014
                                 + 0.99 b_k1_PG_s014 - 1.01 s_k1_PG_s014  = 0
 w_k1_PG_s015_return_constraint: 1.14902 w_k1_PG_s01 - w_k1_PG_s015
                                 + 0.99 b_k1_PG_s015 - 1.01 s_k1_PG_s015  = 0
 w_k1_PG_s016_return_constraint: 1.24233 w_k1_PG_s01 - w_k1_PG_s016
                                 + 0.99 b_k1_PG_s016 - 1.01 s_k1_PG_s016  = 0
 w_k1_PG_s017_return_constraint: 1.20149 w_k1_PG_s01 - w_k1_PG_s017
                                 + 0.99 b_k1_PG_s017 - 1.01 s_k1_PG_s017  = 0
 w_k1_PG_s100_return_constraint: 0.873131 w_k1_PG_s10 - w_k1_PG_s100
                                 + 0.99 b_k1_PG_s100 - 1.01 s_k1_PG_s100  = 0
 w_k1_PG_s101_return_constraint: 1.26994 w_k1_PG_s10 - w_k1_PG_s101
                                 + 0.99 b_k1_PG_s101 - 1.01 s_k1_PG_s101  = 0
 w_k1_PG_s102_return_constraint: 1.07927 w_k1_PG_s10 - w_k1_PG_s102
                                 + 0.99 b_k1_PG_s102 - 1.01 s_k1_PG_s102  = 0
 w_k1_PG_s103_return_constraint: 1.2387 w_k1_PG_s10 - w_k1_PG_s103
                                 + 0.99 b_k1_PG_s103 - 1.01 s_k1_PG_s103  = 0
 w_k1_PG_s104_return_constraint: 0.958928 w_k1_PG_s10 - w_k1_PG_s104
                                 + 0.99 b_k1_PG_s104 - 1.01 s_k1_PG_s104  = 0
 w_k1_PG_s105_return_constraint: 0.786063 w_k1_PG_s10 - w_k1_PG_s105
                                 + 0.99 b_k1_PG_s105 - 1.01 s_k1_PG_s105  = 0
 w_k1_PG_s106_return_constraint: 1.01957 w_k1_PG_s10 - w_k1_PG_s106
                                 + 0.99 b_k1_PG_s106 - 1.01 s_k1_PG_s106  = 0
 w_k1_PG_s107_return_constraint: 0.897435 w_k1_PG_s10 - w_k1_PG_s107
                                 + 0.99 b_k1_PG_s107 - 1.01 s_k1_PG_s107  = 0
 w_k1_PG_s110_return_constraint: 0.815539 w_k1_PG_s11 - w_k1_PG_s110
                                 + 0.99 b_k1_PG_s110 - 1.01 s_k1_PG_s110  = 0
 w_k1_PG_s111_return_constraint: 1.24913 w_k1_PG_s11 - w_k1_PG_s111
                                 + 0.99 b_k1_PG_s111 - 1.01 s_k1_PG_s111  = 0
 w_k1_PG_s112_return_constraint: 0.794488 w_k1_PG_s11 - w_k1_PG_s112
                                 + 0.99 b_k1_PG_s112 - 1.01 s_k1_PG_s112  = 0
 w_k1_PG_s113_return_constraint: 0.962169 w_k1_PG_s11 - w_k1_PG_s113
                                 + 0.99 b_k1_PG_s113 - 1.01 s_k1_PG_s113  = 0
 w_k1_PG_s114_return_constraint: 0.923254 w_k1_PG_s11 - w_k1_PG_s114
                                 + 0.99 b_k1_PG_s114 - 1.01 s_k1_PG_s114  = 0
 w_k1_PG_s115_return_constraint: 1.18108 w_k1_PG_s11 - w_k1_PG_s115
                                 + 0.99 b_k1_PG_s115 - 1.01 s_k1_PG_s115  = 0
 w_k1_PG_s116_return_constraint: 1.05151 w_k1_PG_s11 - w_k1_PG_s116
                                 + 0.99 b_k1_PG_s116 - 1.01 s_k1_PG_s116  = 0
 w_k1_PG_s117_return_constraint: 1.01447 w_k1_PG_s11 - w_k1_PG_s117
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
 k1_wcvar_constraint:            wcvar - var_1 - 0.213000318448533 z_k1_s0000
                                 - 0.225000259358188 z_k1_s0001
                                 - 0.146000275469037 z_k1_s0002
                                 - 0.399001142750929 z_k1_s0003
                                 - 0.507000610937825 z_k1_s0004
                                 - 0.404000496628972 z_k1_s0005
                                 - 0.345999539230347 z_k1_s0006
                                 - 0.24600028025028 z_k1_s0007
                                 - 0.399000993336911 z_k1_s0010
                                 - 0.157000102053206 z_k1_s0011
                                 - 0.434999300847496 z_k1_s0012
                                 - 0.202999930662311 z_k1_s0013
                                 - 0.0920000426184296 z_k1_s0014
                                 - 0.161000170932307 z_k1_s0015
                                 - 0.0949999979277 z_k1_s0016
                                 - 0.384999403359291 z_k1_s0017
                                 - 1.14800056470174 z_k1_s0020
                                 - 0.325000254011837 z_k1_s0021
                                 - 0.330000232207375 z_k1_s0022
                                 - 0.552000378290506 z_k1_s0023
                                 - 1.13900338945295 z_k1_s0024
                                 - 1.05400097462564 z_k1_s0025
                                 - 0.572003076575829 z_k1_s0026
                                 - 0.451000261640015 z_k1_s0027
                                 - 1.11100076387314 z_k1_s0030
                                 - 0.30999984235155 z_k1_s0031
                                 - 0.385999477004584 z_k1_s0032
                                 - 0.572001115973417 z_k1_s0033
                                 - 0.484999319231177 z_k1_s0034
                                 - 0.387999455384941 z_k1_s0035
                                 - 0.839000966947832 z_k1_s0036
                                 - 0.470998558169778 z_k1_s0037
                                 - 0.367999797543009 z_k1_s0040
                                 - 0.313000919405842 z_k1_s0041
                                 - 0.143000155059602 z_k1_s0042
                                 - 0.230000227914723 z_k1_s0043
                                 - 0.252999565435533 z_k1_s0044
                                 - 0.2080002040594 z_k1_s0045
                                 - 0.466000591226477 z_k1_s0046
                                 - 0.382000586069871 z_k1_s0047
                                 - 0.15399998807752 z_k1_s0050
                                 - 0.10600011389373 z_k1_s0051
                                 - 0.274999978709857 z_k1_s0052
                                 - 0.104000114048568 z_k1_s0053
                                 - 0.144999988774288 z_k1_s0054
                                 - 0.0920001149775921 z_k1_s0055
                                 - 0.171999986683983 z_k1_s0056
                                 - 0.172999986606565 z_k1_s0057
                                 - 0.321999509042984 z_k1_s0060
                                 - 0.527997857901136 z_k1_s0061
                                 - 0.368998729464646 z_k1_s0062
                                 - 0.266999596278494 z_k1_s0063
                                 - 1.08499854264453 z_k1_s0064
                                 - 0.479998052637397 z_k1_s0065
                                 - 0.248999578704732 z_k1_s0066
                                 - 0.323999591528832 z_k1_s0067
                                 - 0.571999638639825 z_k1_s0070
                                 - 0.153000014590091 z_k1_s0071
                                 - 0.205000481666745 z_k1_s0072
                                 - 0.179000051728298 z_k1_s0073
                                 - 0.196999140761867 z_k1_s0074
                                 - 0.23500060005703 z_k1_s0075
                                 - 0.173000028050241 z_k1_s0076
                                 - 0.249999677251571 z_k1_s0077
                                 - 0.228999542635642 z_k1_s0100
                                 - 0.225999399101918 z_k1_s0101
                                 - 0.789997513312962 z_k1_s0102
                                 - 0.228999542635642 z_k1_s0103
                                 - 0.462999660193333 z_k1_s0104
                                 - 0.843998082724943 z_k1_s0105
                                 - 0.369997560542112 z_k1_s0106
                                 - 0.20599944930795 z_k1_s0107
                                 - 0.152999344878646 z_k1_s0110
                                 - 0.388997411797715 z_k1_s0111
                                 - 0.319998496106193 z_k1_s0112
                                 - 0.381998872417883 z_k1_s0113
                                 - 0.143999436897338 z_k1_s0114
                                 - 0.207999136118577 z_k1_s0115
                                 - 0.409997575919272 z_k1_s0116
                                 - 0.266999164271326 z_k1_s0117
                                 - 0.276998803955783 z_k1_s0120
                                 - 0.163999303723588 z_k1_s0121
                                 - 0.244998412324226 z_k1_s0122
                                 - 0.189999139176346 z_k1_s0123
                                 - 0.0869996647794367 z_k1_s0124
                                 - 0.413998791010798 z_k1_s0125
                                 - 0.425997972376867 z_k1_s0126
                                 - 0.127999442435546 z_k1_s0127
                                 - 0.22899942515462 z_k1_s0130
                                 - 0.253000653782427 z_k1_s0131
                                 - 0.144999988258233 z_k1_s0132
                                 - 0.134000040037274 z_k1_s0133
                                 - 0.379999888269837 z_k1_s0134
                                 - 0.090000079453403 z_k1_s0135
                                 - 0.309000725580104 z_k1_s0136
                                 - 0.137000025915717 z_k1_s0137
                                 - 0.336001549603955 z_k1_s0140
                                 - 0.268999936138274 z_k1_s0141
                                 - 0.303999732304516 z_k1_s0142
                                 - 0.127000345691421 z_k1_s0143
                                 - 0.222000432486973 z_k1_s0144
                                 - 0.20100094438812 z_k1_s0145
                                 - 0.26400135668914 z_k1_s0146
                                 - 0.225000359358238 z_k1_s0147
                                 - 0.194999480252083 z_k1_s0150
                                 - 0.141999816003831 z_k1_s0151
                                 - 0.105999935295724 z_k1_s0152
                                 - 0.216998985963135 z_k1_s0153
                                 - 0.139999805775625 z_k1_s0154
                                 - 0.294000289947914 z_k1_s0155
                                 - 0.250999463242255 z_k1_s0156
                                 - 0.172000272826538 z_k1_s0157
                                 - 0.152000104941025 z_k1_s0160
                                 - 0.166999836810727 z_k1_s0161
                                 - 0.0489998711062934 z_k1_s0162
                                 - 0.0609999056015883 z_k1_s0163
                                 - 0.149999808692354 z_k1_s0164
                                 - 0.101000020585905 z_k1_s0165
                                 - 0.0479998889816466 z_k1_s0166
                                 - 0.10200016871024 z_k1_s0167
                                 - 0.149999712155438 z_k1_s0170
                                 - 0.257998633569226 z_k1_s0171
                                 - 0.217999627525805 z_k1_s0172
                                 - 0.287000435245286 z_k1_s0173
                                 - 0.525999311644379 z_k1_s0174
                                 - 0.209999597017613 z_k1_s0175
                                 - 0.125999620630864 z_k1_s0176
                                 - 0.517999051836681 z_k1_s0177
                                 - 0.524998587252629 z_k1_s1000
                                 - 0.57800020737315 z_k1_s1001
                                 - 0.135999895687985 z_k1_s1002
                                 - 0.190999885399999 z_k1_s1003
                                 - 0.288999525287272 z_k1_s1004
                                 - 0.465998966361115 z_k1_s1005
                                 - 0.341998831810583 z_k1_s1006
                                 - 0.364998880090089 z_k1_s1007
                                 - 0.297999811045532 z_k1_s1010
                                 - 0.236999902250279 z_k1_s1011
                                 - 0.196999844446894 z_k1_s1012
                                 - 0.254999960871786 z_k1_s1013
                                 - 0.271999871303282 z_k1_s1014
                                 - 0.944000320640413 z_k1_s1015
                                 - 0.580001164248913 z_k1_s1016
                                 - 0.478000397260598 z_k1_s1017
                                 - 0.341000731136018 z_k1_s1020
                                 - 0.456000782880961 z_k1_s1021
                                 - 1.01300449971254 z_k1_s1022
                                 - 1.38400408894521 z_k1_s1023
                                 - 0.462000785580698 z_k1_s1024
                                 - 0.362000740585095 z_k1_s1025
                                 - 0.867002123214912 z_k1_s1026
                                 - 0.891999245958759 z_k1_s1027
                                 - 0.799997151159014 z_k1_s1030
                                 - 0.384998688545103 z_k1_s1031
                                 - 0.405998816232442 z_k1_s1032
                                 - 0.764996144349078 z_k1_s1033
                                 - 1.17999651255748 z_k1_s1034
                                 - 0.411998580486754 z_k1_s1035
                                 - 0.498998973363266 z_k1_s1036
                                 - 0.316998978336449 z_k1_s1037
                                 - 0.604999625393989 z_k1_s1040
                                 - 0.514000100971285 z_k1_s1041
                                 - 1.78000027849591 z_k1_s1042
                                 - 0.565999567912915 z_k1_s1043
                                 - 0.423999874522685 z_k1_s1044
                                 - 0.402999984271292 z_k1_s1045
                                 - 0.511999675923676 z_k1_s1046
                                 - 1.29300256231845 z_k1_s1047
                                 - 0.841001400034155 z_k1_s1050
                                 - 1.40699904342278 z_k1_s1051
                                 - 1.01400286776534 z_k1_s1052
                                 - 1.0089979526313 z_k1_s1053
                                 - 1.7010002927517 z_k1_s1054
                                 - 1.54100189465713 z_k1_s1055
                                 - 0.664000413701149 z_k1_s1056
                                 - 0.650000774543144 z_k1_s1057
                                 - 0.39299868990054 z_k1_s1060
                                 - 0.429999908633044 z_k1_s1061
                                 - 0.405998899914847 z_k1_s1062
                                 - 0.18899980338768 z_k1_s1063
                                 - 0.365999909715578 z_k1_s1064
                                 - 0.316999532122838 z_k1_s1065
                                 - 0.17899978673809 z_k1_s1066
                                 - 0.410998773689756 z_k1_s1067
                                 - 0.489998727451863 z_k1_s1070
                                 - 0.690998061881457 z_k1_s1071
                                 - 0.76999958706323 z_k1_s1072
                                 - 1.1589980705569 z_k1_s1073
                                 - 1.13299918082214 z_k1_s1074
                                 - 1.41499568292383 z_k1_s1075
                                 - 0.941995090301398 z_k1_s1076
                                 - 0.804995529212607 z_k1_s1077
                                 - 0.407000270904594 z_k1_s1100
                                 - 0.256000251883358 z_k1_s1101
                                 - 0.153999678232103 z_k1_s1102
                                 - 0.493998568707473 z_k1_s1103
                                 - 0.193999703770148 z_k1_s1104
                                 - 0.171999844493992 z_k1_s1105
                                 - 0.384998642831085 z_k1_s1106
                                 - 0.148999730314765 z_k1_s1107
                                 - 0.213000612621341 z_k1_s1110
                                 - 0.165000300734733 z_k1_s1111
                                 - 0.135000246055691 z_k1_s1112
                                 - 0.143999813659124 z_k1_s1113
                                 - 0.112000017134975 z_k1_s1114
                                 - 0.0950000235502078 z_k1_s1115
                                 - 0.117999691470574 z_k1_s1116
                                 - 0.139999881168632 z_k1_s1117
                                 - 0.218999693379202 z_k1_s1120
                                 - 0.0610000826697263 z_k1_s1121
                                 - 0.158000305210207 z_k1_s1122
                                 - 0.29700045943299 z_k1_s1123
                                 - 0.176000375145584 z_k1_s1124
                                 - 0.264000562718375 z_k1_s1125
                                 - 0.142000397378923 z_k1_s1126
                                 - 0.0720001408413618 z_k1_s1127
                                 - 0.143000060536289 z_k1_s1130
                                 - 0.212999711879618 z_k1_s1131
                                 - 0.160000836614657 z_k1_s1132
                                 - 0.207000834982511 z_k1_s1133
                                 - 0.327000362942607 z_k1_s1134
                                 - 0.196999774818272 z_k1_s1135
                                 - 0.129000042307551 z_k1_s1136
                                 - 0.0900000124703709 z_k1_s1137
                                 - 0.123000582795855 z_k1_s1140
                                 - 0.109000674335409 z_k1_s1141
                                 - 0.225002033705074 z_k1_s1142
                                 - 0.246001165591709 z_k1_s1143
                                 - 0.412002257691866 z_k1_s1144
                                 - 0.480002528960838 z_k1_s1145
                                 - 0.334002499239331 z_k1_s1146
                                 - 0.159000824668206 z_k1_s1147
                                 - 0.508002698824047 z_k1_s1150
                                 - 0.334000927355064 z_k1_s1151
                                 - 0.185000431259899 z_k1_s1152
                                 - 0.38800172360478 z_k1_s1153
                                 - 0.459003015411605 z_k1_s1154
                                 - 0.332000911101409 z_k1_s1155
                                 - 0.75000252110996 z_k1_s1156
                                 - 0.618003592775042 z_k1_s1157
                                 - 0.35700157332988 z_k1_s1160
                                 - 0.185001011732275 z_k1_s1161
                                 - 0.398003023985554 z_k1_s1162
                                 - 0.112000576564743 z_k1_s1163
                                 - 0.0980004793065082 z_k1_s1164
                                 - 0.180000962604253 z_k1_s1165
                                 - 0.354001221451253 z_k1_s1166
                                 - 0.33100188206734 z_k1_s1167
                                 - 0.612001638175039 z_k1_s1170
                                 - 0.84100174910692 z_k1_s1171
                                 - 0.33400072190686 z_k1_s1172
                                 - 0.436000994936033 z_k1_s1173
                                 - 0.466000960485501 z_k1_s1174
                                 - 0.411000942361272 z_k1_s1175
                                 - 0.316000645036934 z_k1_s1176
                                 - 1.46100168739757 z_k1_s1177 >= 0
 z_k1_s0000_constraint:          var_1 + z_k1_s0000 + 1.35171 w_k1_ADM_s000
                                 + 0.944562 w_k1_AXP_s000
                                 + 1.30865 w_k1_DIS_s000
                                 + 0.97962 w_k1_FDX_s000
                                 + 1.21017 w_k1_IBM_s000
                                 + 1.17386 w_k1_INTC_s000
                                 + 1.25188 w_k1_KO_s000
                                 + 0.759015 w_k1_MAT_s000
                                 + 1.1889 w_k1_MSFT_s000 + 1.07779 w_k1_PG_s000
                                 >= 1
 z_k1_s0001_constraint:          var_1 + z_k1_s0001 + 1.51725 w_k1_ADM_s000
                                 + 1.44353 w_k1_AXP_s000
                                 + 1.34004 w_k1_DIS_s000
                                 + 1.13145 w_k1_FDX_s000
                                 + 1.54508 w_k1_IBM_s000
                                 + 1.08201 w_k1_INTC_s000
                                 + 1.48161 w_k1_KO_s000 + 1.39067 w_k1_MAT_s000
                                 + 1.2825 w_k1_MSFT_s000 + 1.19417 w_k1_PG_s000
                                 >= 1
 z_k1_s0002_constraint:          var_1 + z_k1_s0002 + 1.18094 w_k1_ADM_s000
                                 + 1.19018 w_k1_AXP_s000
                                 + 0.934143 w_k1_DIS_s000
                                 + 0.997249 w_k1_FDX_s000
                                 + 1.25638 w_k1_IBM_s000
                                 + 0.917034 w_k1_INTC_s000
                                 + 0.879197 w_k1_KO_s000 + 1.3885 w_k1_MAT_s000
                                 + 1.0354 w_k1_MSFT_s000 + 1.10356 w_k1_PG_s000
                                 >= 1
 z_k1_s0003_constraint:          var_1 + z_k1_s0003 + 0.771943 w_k1_ADM_s000
                                 + 0.693149 w_k1_AXP_s000
                                 + 0.696417 w_k1_DIS_s000
                                 + 0.807605 w_k1_FDX_s000
                                 + 0.707828 w_k1_IBM_s000
                                 + 0.938168 w_k1_INTC_s000
                                 + 0.648851 w_k1_KO_s000
                                 + 0.502967 w_k1_MAT_s000
                                 + 0.930275 w_k1_MSFT_s000
                                 + 0.886411 w_k1_PG_s000 >= 1
 z_k1_s0004_constraint:          var_1 + z_k1_s0004 + 1.10041 w_k1_ADM_s000
                                 + 1.01873 w_k1_AXP_s000 + 0.8427 w_k1_DIS_s000
                                 + 0.980744 w_k1_FDX_s000
                                 + 0.782105 w_k1_IBM_s000
                                 + 1.18131 w_k1_INTC_s000
                                 + 0.804305 w_k1_KO_s000
                                 + 0.695701 w_k1_MAT_s000
                                 + 0.995851 w_k1_MSFT_s000
                                 + 0.961754 w_k1_PG_s000 >= 1
 z_k1_s0005_constraint:          var_1 + z_k1_s0005 + 1.11868 w_k1_ADM_s000
                                 + 1.24181 w_k1_AXP_s000
                                 + 1.30809 w_k1_DIS_s000
                                 + 1.22267 w_k1_FDX_s000
                                 + 0.876922 w_k1_IBM_s000
                                 + 1.17772 w_k1_INTC_s000
                                 + 0.992793 w_k1_KO_s000
                                 + 1.06354 w_k1_MAT_s000
                                 + 1.22266 w_k1_MSFT_s000 + 1.059 w_k1_PG_s000
                                 >= 1
 z_k1_s0006_constraint:          var_1 + z_k1_s0006 + 1.07961 w_k1_ADM_s000
                                 + 0.932608 w_k1_AXP_s000
                                 + 0.909162 w_k1_DIS_s000
                                 + 0.800257 w_k1_FDX_s000
                                 + 1.1704 w_k1_IBM_s000
                                 + 1.06691 w_k1_INTC_s000
                                 + 0.905145 w_k1_KO_s000
                                 + 0.874573 w_k1_MAT_s000
                                 + 0.886211 w_k1_MSFT_s000
                                 + 1.03782 w_k1_PG_s000 >= 1
 z_k1_s0007_constraint:          var_1 + z_k1_s0007 + 1.07674 w_k1_ADM_s000
                                 + 1.5402 w_k1_AXP_s000 + 1.13897 w_k1_DIS_s000
                                 + 1.1389 w_k1_FDX_s000 + 1.0613 w_k1_IBM_s000
                                 + 1.38653 w_k1_INTC_s000
                                 + 0.884575 w_k1_KO_s000
                                 + 1.04059 w_k1_MAT_s000
                                 + 0.987151 w_k1_MSFT_s000
                                 + 1.0716 w_k1_PG_s000 >= 1
 z_k1_s0010_constraint:          var_1 + z_k1_s0010 + 0.919723 w_k1_ADM_s001
                                 + 1.01828 w_k1_AXP_s001
                                 + 0.750517 w_k1_DIS_s001
                                 + 0.91799 w_k1_FDX_s001
                                 + 0.860738 w_k1_IBM_s001
                                 + 0.955735 w_k1_INTC_s001
                                 + 0.784997 w_k1_KO_s001
                                 + 0.983509 w_k1_MAT_s001
                                 + 0.919344 w_k1_MSFT_s001
                                 + 0.818975 w_k1_PG_s001 >= 1
 z_k1_s0011_constraint:          var_1 + z_k1_s0011 + 0.666162 w_k1_ADM_s001
                                 + 0.686629 w_k1_AXP_s001
                                 + 0.703621 w_k1_DIS_s001
                                 + 0.755217 w_k1_FDX_s001
                                 + 0.659767 w_k1_IBM_s001
                                 + 0.866438 w_k1_INTC_s001
                                 + 0.670441 w_k1_KO_s001
                                 + 0.711792 w_k1_MAT_s001
                                 + 0.857202 w_k1_MSFT_s001
                                 + 0.715124 w_k1_PG_s001 >= 1
 z_k1_s0012_constraint:          var_1 + z_k1_s0012 + 0.983117 w_k1_ADM_s001
                                 + 0.88006 w_k1_AXP_s001
                                 + 0.957114 w_k1_DIS_s001
                                 + 0.877764 w_k1_FDX_s001
                                 + 1.10762 w_k1_IBM_s001
                                 + 1.14413 w_k1_INTC_s001
                                 + 1.03143 w_k1_KO_s001 + 1.08362 w_k1_MAT_s001
                                 + 0.985188 w_k1_MSFT_s001
                                 + 1.01124 w_k1_PG_s001 >= 1
 z_k1_s0013_constraint:          var_1 + z_k1_s0013 + 1.10485 w_k1_ADM_s001
                                 + 1.01846 w_k1_AXP_s001
                                 + 1.45244 w_k1_DIS_s001
                                 + 1.13416 w_k1_FDX_s001
                                 + 1.00003 w_k1_IBM_s001
                                 + 1.31906 w_k1_INTC_s001
                                 + 1.17952 w_k1_KO_s001
                                 + 0.952673 w_k1_MAT_s001
                                 + 1.46945 w_k1_MSFT_s001
                                 + 1.23266 w_k1_PG_s001 >= 1
 z_k1_s0014_constraint:          var_1 + z_k1_s0014 + 0.911271 w_k1_ADM_s001
                                 + 0.796139 w_k1_AXP_s001
                                 + 1.07843 w_k1_DIS_s001
                                 + 1.01365 w_k1_FDX_s001
                                 + 0.717677 w_k1_IBM_s001
                                 + 1.0173 w_k1_INTC_s001 + 1.00638 w_k1_KO_s001
                                 + 0.842523 w_k1_MAT_s001
                                 + 1.19616 w_k1_MSFT_s001
                                 + 0.962836 w_k1_PG_s001 >= 1
 z_k1_s0015_constraint:          var_1 + z_k1_s0015 + 1.32419 w_k1_ADM_s001
                                 + 1.2583 w_k1_AXP_s001
                                 + 0.961762 w_k1_DIS_s001
                                 + 1.20716 w_k1_FDX_s001
                                 + 0.752529 w_k1_IBM_s001
                                 + 1.03528 w_k1_INTC_s001
                                 + 0.748997 w_k1_KO_s001
                                 + 0.83961 w_k1_MAT_s001
                                 + 1.21475 w_k1_MSFT_s001
                                 + 1.02232 w_k1_PG_s001 >= 1
 z_k1_s0016_constraint:          var_1 + z_k1_s0016 + 1.02564 w_k1_ADM_s001
                                 + 0.734492 w_k1_AXP_s001
                                 + 0.815943 w_k1_DIS_s001
                                 + 0.939462 w_k1_FDX_s001
                                 + 0.617904 w_k1_IBM_s001
                                 + 0.978865 w_k1_INTC_s001
                                 + 0.766732 w_k1_KO_s001
                                 + 0.733797 w_k1_MAT_s001
                                 + 1.03632 w_k1_MSFT_s001
                                 + 0.825621 w_k1_PG_s001 >= 1
 z_k1_s0017_constraint:          var_1 + z_k1_s0017 + 1.3098 w_k1_ADM_s001
                                 + 1.62681 w_k1_AXP_s001
                                 + 1.40336 w_k1_DIS_s001 + 1.1986 w_k1_FDX_s001
                                 + 1.24307 w_k1_IBM_s001
                                 + 1.27219 w_k1_INTC_s001
                                 + 1.33594 w_k1_KO_s001 + 1.62464 w_k1_MAT_s001
                                 + 1.32533 w_k1_MSFT_s001
                                 + 1.21945 w_k1_PG_s001 >= 1
 z_k1_s0020_constraint:          var_1 + z_k1_s0020 + 1.03573 w_k1_ADM_s002
                                 + 1.05629 w_k1_AXP_s002
                                 + 1.00375 w_k1_DIS_s002
                                 + 0.998866 w_k1_FDX_s002
                                 + 1.02624 w_k1_IBM_s002
                                 + 1.17627 w_k1_INTC_s002
                                 + 1.06406 w_k1_KO_s002
                                 + 0.774546 w_k1_MAT_s002
                                 + 1.09402 w_k1_MSFT_s002
                                 + 1.07108 w_k1_PG_s002 >= 1
 z_k1_s0021_constraint:          var_1 + z_k1_s0021 + 0.81961 w_k1_ADM_s002
                                 + 0.925656 w_k1_AXP_s002
                                 + 0.87956 w_k1_DIS_s002
                                 + 0.827888 w_k1_FDX_s002
                                 + 1.35979 w_k1_IBM_s002
                                 + 1.16875 w_k1_INTC_s002
                                 + 0.971787 w_k1_KO_s002
                                 + 0.946549 w_k1_MAT_s002
                                 + 0.9518 w_k1_MSFT_s002
                                 + 0.926578 w_k1_PG_s002 >= 1
 z_k1_s0022_constraint:          var_1 + z_k1_s0022 + 0.891584 w_k1_ADM_s002
                                 + 0.768803 w_k1_AXP_s002
                                 + 0.740465 w_k1_DIS_s002
                                 + 0.946216 w_k1_FDX_s002
                                 + 0.492468 w_k1_IBM_s002
                                 + 1.20524 w_k1_INTC_s002
                                 + 0.651933 w_k1_KO_s002
                                 + 0.463921 w_k1_MAT_s002
                                 + 1.07089 w_k1_MSFT_s002
                                 + 0.792805 w_k1_PG_s002 >= 1
 z_k1_s0023_constraint:          var_1 + z_k1_s0023 + 0.919178 w_k1_ADM_s002
                                 + 1.25699 w_k1_AXP_s002
                                 + 1.17212 w_k1_DIS_s002
                                 + 0.968689 w_k1_FDX_s002
                                 + 1.62728 w_k1_IBM_s002
                                 + 1.62856 w_k1_INTC_s002
                                 + 1.23909 w_k1_KO_s002 + 1.16174 w_k1_MAT_s002
                                 + 1.17572 w_k1_MSFT_s002
                                 + 1.04982 w_k1_PG_s002 >= 1
 z_k1_s0024_constraint:          var_1 + z_k1_s0024 + 0.910869 w_k1_ADM_s002
                                 + 0.761686 w_k1_AXP_s002
                                 + 0.706678 w_k1_DIS_s002
                                 + 0.825741 w_k1_FDX_s002
                                 + 0.684003 w_k1_IBM_s002
                                 + 1.00845 w_k1_INTC_s002
                                 + 0.682388 w_k1_KO_s002
                                 + 0.509807 w_k1_MAT_s002
                                 + 0.945599 w_k1_MSFT_s002
                                 + 0.790875 w_k1_PG_s002 >= 1
 z_k1_s0025_constraint:          var_1 + z_k1_s0025 + 1.3225 w_k1_ADM_s002
                                 + 1.21176 w_k1_AXP_s002
                                 + 1.37451 w_k1_DIS_s002 + 1.2545 w_k1_FDX_s002
                                 + 0.943842 w_k1_IBM_s002
                                 + 1.21438 w_k1_INTC_s002
                                 + 0.942008 w_k1_KO_s002
                                 + 0.816144 w_k1_MAT_s002
                                 + 1.23123 w_k1_MSFT_s002
                                 + 1.07569 w_k1_PG_s002 >= 1
 z_k1_s0026_constraint:          var_1 + z_k1_s0026 + 0.720989 w_k1_ADM_s002
                                 + 0.703128 w_k1_AXP_s002
                                 + 0.698269 w_k1_DIS_s002
                                 + 0.737655 w_k1_FDX_s002
                                 + 0.917867 w_k1_IBM_s002
                                 + 1.02992 w_k1_INTC_s002
                                 + 0.849531 w_k1_KO_s002
                                 + 0.765194 w_k1_MAT_s002
                                 + 0.865113 w_k1_MSFT_s002
                                 + 0.834961 w_k1_PG_s002 >= 1
 z_k1_s0027_constraint:          var_1 + z_k1_s0027 + 1.19958 w_k1_ADM_s002
                                 + 1.48309 w_k1_AXP_s002
                                 + 1.83968 w_k1_DIS_s002
                                 + 1.30877 w_k1_FDX_s002
                                 + 1.60526 w_k1_IBM_s002 + 1.549 w_k1_INTC_s002
                                 + 1.43823 w_k1_KO_s002 + 1.70569 w_k1_MAT_s002
                                 + 1.46002 w_k1_MSFT_s002
                                 + 1.13342 w_k1_PG_s002 >= 1
 z_k1_s0030_constraint:          var_1 + z_k1_s0030 + 1.0469 w_k1_ADM_s003
                                 + 0.724213 w_k1_AXP_s003
                                 + 0.740859 w_k1_DIS_s003
                                 + 0.869128 w_k1_FDX_s003
                                 + 0.791449 w_k1_IBM_s003
                                 + 0.867543 w_k1_INTC_s003
                                 + 0.813973 w_k1_KO_s003
                                 + 0.650142 w_k1_MAT_s003
                                 + 0.708725 w_k1_MSFT_s003
                                 + 0.951456 w_k1_PG_s003 >= 1
 z_k1_s0031_constraint:          var_1 + z_k1_s0031 + 1.24905 w_k1_ADM_s003
                                 + 1.01432 w_k1_AXP_s003
                                 + 1.31125 w_k1_DIS_s003
                                 + 1.33813 w_k1_FDX_s003
                                 + 0.875909 w_k1_IBM_s003
                                 + 1.11551 w_k1_INTC_s003 + 1.1626 w_k1_KO_s003
                                 + 0.923122 w_k1_MAT_s003
                                 + 1.07512 w_k1_MSFT_s003
                                 + 1.18866 w_k1_PG_s003 >= 1
 z_k1_s0032_constraint:          var_1 + z_k1_s0032 + 1.06843 w_k1_ADM_s003
                                 + 1.0158 w_k1_AXP_s003
                                 + 0.774049 w_k1_DIS_s003
                                 + 1.04467 w_k1_FDX_s003
                                 + 0.587861 w_k1_IBM_s003
                                 + 0.845165 w_k1_INTC_s003
                                 + 0.696821 w_k1_KO_s003
                                 + 0.700881 w_k1_MAT_s003
                                 + 0.887438 w_k1_MSFT_s003
                                 + 0.880811 w_k1_PG_s003 >= 1
 z_k1_s0033_constraint:          var_1 + z_k1_s0033 + 0.892817 w_k1_ADM_s003
                                 + 1.00354 w_k1_AXP_s003
                                 + 0.783789 w_k1_DIS_s003
                                 + 0.805172 w_k1_FDX_s003
                                 + 1.39301 w_k1_IBM_s003
                                 + 1.02357 w_k1_INTC_s003
                                 + 1.11348 w_k1_KO_s003 + 1.3765 w_k1_MAT_s003
                                 + 0.839729 w_k1_MSFT_s003
                                 + 1.08411 w_k1_PG_s003 >= 1
 z_k1_s0034_constraint:          var_1 + z_k1_s0034 + 1.13131 w_k1_ADM_s003
                                 + 0.767853 w_k1_AXP_s003
                                 + 0.998499 w_k1_DIS_s003
                                 + 0.926655 w_k1_FDX_s003
                                 + 1.13628 w_k1_IBM_s003
                                 + 0.876743 w_k1_INTC_s003
                                 + 1.00387 w_k1_KO_s003
                                 + 0.603513 w_k1_MAT_s003
                                 + 0.866951 w_k1_MSFT_s003
                                 + 0.97236 w_k1_PG_s003 >= 1
 z_k1_s0035_constraint:          var_1 + z_k1_s0035 + 1.11683 w_k1_ADM_s003
                                 + 1.03657 w_k1_AXP_s003
                                 + 0.919085 w_k1_DIS_s003
                                 + 0.964832 w_k1_FDX_s003
                                 + 0.862992 w_k1_IBM_s003
                                 + 0.815158 w_k1_INTC_s003
                                 + 1.05272 w_k1_KO_s003 + 1.141 w_k1_MAT_s003
                                 + 0.85894 w_k1_MSFT_s003
                                 + 1.06967 w_k1_PG_s003 >= 1
 z_k1_s0036_constraint:          var_1 + z_k1_s0036 + 1.03237 w_k1_ADM_s003
                                 + 1.30081 w_k1_AXP_s003
                                 + 1.52265 w_k1_DIS_s003
                                 + 1.31432 w_k1_FDX_s003
                                 + 1.51414 w_k1_IBM_s003
                                 + 1.24112 w_k1_INTC_s003
                                 + 1.38881 w_k1_KO_s003 + 1.3301 w_k1_MAT_s003
                                 + 1.24309 w_k1_MSFT_s003
                                 + 1.33311 w_k1_PG_s003 >= 1
 z_k1_s0037_constraint:          var_1 + z_k1_s0037 + 0.953622 w_k1_ADM_s003
                                 + 1.08 w_k1_AXP_s003 + 1.12781 w_k1_DIS_s003
                                 + 1.16906 w_k1_FDX_s003
                                 + 1.01114 w_k1_IBM_s003
                                 + 1.03545 w_k1_INTC_s003
                                 + 1.08967 w_k1_KO_s003
                                 + 0.959513 w_k1_MAT_s003
                                 + 1.0367 w_k1_MSFT_s003 + 1.11767 w_k1_PG_s003
                                 >= 1
 z_k1_s0040_constraint:          var_1 + z_k1_s0040 + 1.32134 w_k1_ADM_s004
                                 + 1.23735 w_k1_AXP_s004
                                 + 1.10322 w_k1_DIS_s004 + 1.0836 w_k1_FDX_s004
                                 + 1.15446 w_k1_IBM_s004
                                 + 0.887278 w_k1_INTC_s004
                                 + 1.04759 w_k1_KO_s004 + 1.12717 w_k1_MAT_s004
                                 + 1.01945 w_k1_MSFT_s004
                                 + 1.13917 w_k1_PG_s004 >= 1
 z_k1_s0041_constraint:          var_1 + z_k1_s0041 + 0.933566 w_k1_ADM_s004
                                 + 1.01289 w_k1_AXP_s004
                                 + 0.747425 w_k1_DIS_s004
                                 + 0.783662 w_k1_FDX_s004
                                 + 1.18508 w_k1_IBM_s004
                                 + 0.843711 w_k1_INTC_s004
                                 + 1.08924 w_k1_KO_s004 + 1.24877 w_k1_MAT_s004
                                 + 0.735175 w_k1_MSFT_s004
                                 + 0.859751 w_k1_PG_s004 >= 1
 z_k1_s0042_constraint:          var_1 + z_k1_s0042 + 1.01382 w_k1_ADM_s004
                                 + 0.977363 w_k1_AXP_s004
                                 + 0.845783 w_k1_DIS_s004
                                 + 0.883049 w_k1_FDX_s004
                                 + 1.07611 w_k1_IBM_s004
                                 + 1.0194 w_k1_INTC_s004 + 1.08794 w_k1_KO_s004
                                 + 0.947821 w_k1_MAT_s004
                                 + 1.00765 w_k1_MSFT_s004
                                 + 1.05748 w_k1_PG_s004 >= 1
 z_k1_s0043_constraint:          var_1 + z_k1_s0043 + 1.15684 w_k1_ADM_s004
                                 + 1.00781 w_k1_AXP_s004
                                 + 1.35109 w_k1_DIS_s004
                                 + 1.19805 w_k1_FDX_s004
                                 + 1.23652 w_k1_IBM_s004
                                 + 1.06082 w_k1_INTC_s004
                                 + 1.49522 w_k1_KO_s004 + 1.12215 w_k1_MAT_s004
                                 + 1.1401 w_k1_MSFT_s004 + 1.14989 w_k1_PG_s004
                                 >= 1
 z_k1_s0044_constraint:          var_1 + z_k1_s0044 + 1.64253 w_k1_ADM_s004
                                 + 1.64729 w_k1_AXP_s004 + 1.5772 w_k1_DIS_s004
                                 + 1.26711 w_k1_FDX_s004
                                 + 1.53191 w_k1_IBM_s004
                                 + 1.31035 w_k1_INTC_s004 + 1.6264 w_k1_KO_s004
                                 + 1.43463 w_k1_MAT_s004
                                 + 1.23551 w_k1_MSFT_s004
                                 + 1.50039 w_k1_PG_s004 >= 1
 z_k1_s0045_constraint:          var_1 + z_k1_s0045 + 1.34317 w_k1_ADM_s004
                                 + 1.21051 w_k1_AXP_s004
                                 + 1.08259 w_k1_DIS_s004
                                 + 1.15787 w_k1_FDX_s004
                                 + 0.936894 w_k1_IBM_s004
                                 + 1.24002 w_k1_INTC_s004
                                 + 1.22862 w_k1_KO_s004
                                 + 0.970513 w_k1_MAT_s004
                                 + 1.22861 w_k1_MSFT_s004
                                 + 1.08339 w_k1_PG_s004 >= 1
 z_k1_s0046_constraint:          var_1 + z_k1_s0046 + 0.899956 w_k1_ADM_s004
                                 + 0.802252 w_k1_AXP_s004
                                 + 0.782289 w_k1_DIS_s004
                                 + 0.862783 w_k1_FDX_s004
                                 + 0.758019 w_k1_IBM_s004
                                 + 0.788692 w_k1_INTC_s004
                                 + 0.90277 w_k1_KO_s004
                                 + 0.586709 w_k1_MAT_s004
                                 + 0.877571 w_k1_MSFT_s004
                                 + 0.791745 w_k1_PG_s004 >= 1
 z_k1_s0047_constraint:          var_1 + z_k1_s0047 + 1.20809 w_k1_ADM_s004
                                 + 1.14325 w_k1_AXP_s004
                                 + 0.95025 w_k1_DIS_s004
                                 + 1.01222 w_k1_FDX_s004
                                 + 0.781835 w_k1_IBM_s004
                                 + 0.853374 w_k1_INTC_s004
                                 + 1.01801 w_k1_KO_s004 + 1.00502 w_k1_MAT_s004
                                 + 0.984699 w_k1_MSFT_s004
                                 + 0.880959 w_k1_PG_s004 >= 1
 z_k1_s0050_constraint:          var_1 + z_k1_s0050 + 1.14579 w_k1_ADM_s005
                                 + 0.90439 w_k1_AXP_s005
                                 + 0.918734 w_k1_DIS_s005
                                 + 1.01177 w_k1_FDX_s005
                                 + 0.794125 w_k1_IBM_s005
                                 + 1.01497 w_k1_INTC_s005
                                 + 0.858374 w_k1_KO_s005
                                 + 0.665362 w_k1_MAT_s005
                                 + 1.0233 w_k1_MSFT_s005
                                 + 0.847498 w_k1_PG_s005 >= 1
 z_k1_s0051_constraint:          var_1 + z_k1_s0051 + 0.975035 w_k1_ADM_s005
                                 + 0.921574 w_k1_AXP_s005
                                 + 1.14756 w_k1_DIS_s005
                                 + 1.03505 w_k1_FDX_s005 + 1.3758 w_k1_IBM_s005
                                 + 0.985226 w_k1_INTC_s005
                                 + 1.17998 w_k1_KO_s005 + 1.32196 w_k1_MAT_s005
                                 + 1.02376 w_k1_MSFT_s005
                                 + 0.952499 w_k1_PG_s005 >= 1
 z_k1_s0052_constraint:          var_1 + z_k1_s0052 + 1.05026 w_k1_ADM_s005
                                 + 0.903542 w_k1_AXP_s005
                                 + 1.04505 w_k1_DIS_s005 + 1.0118 w_k1_FDX_s005
                                 + 1.09283 w_k1_IBM_s005
                                 + 1.01731 w_k1_INTC_s005
                                 + 1.09708 w_k1_KO_s005
                                 + 0.950106 w_k1_MAT_s005
                                 + 1.06595 w_k1_MSFT_s005
                                 + 0.875722 w_k1_PG_s005 >= 1
 z_k1_s0053_constraint:          var_1 + z_k1_s0053 + 0.745717 w_k1_ADM_s005
                                 + 0.587191 w_k1_AXP_s005
                                 + 0.61679 w_k1_DIS_s005
                                 + 0.759143 w_k1_FDX_s005
                                 + 0.702214 w_k1_IBM_s005
                                 + 0.959062 w_k1_INTC_s005
                                 + 0.658733 w_k1_KO_s005
                                 + 0.479272 w_k1_MAT_s005
                                 + 0.796612 w_k1_MSFT_s005
                                 + 0.652023 w_k1_PG_s005 >= 1
 z_k1_s0054_constraint:          var_1 + z_k1_s0054 + 0.904712 w_k1_ADM_s005
                                 + 0.736017 w_k1_AXP_s005
                                 + 0.691383 w_k1_DIS_s005
                                 + 0.777119 w_k1_FDX_s005
                                 + 1.06062 w_k1_IBM_s005
                                 + 0.796241 w_k1_INTC_s005
                                 + 0.774968 w_k1_KO_s005
                                 + 0.91653 w_k1_MAT_s005
                                 + 0.841963 w_k1_MSFT_s005
                                 + 0.811758 w_k1_PG_s005 >= 1
 z_k1_s0055_constraint:          var_1 + z_k1_s0055 + 1.00765 w_k1_ADM_s005
                                 + 0.683081 w_k1_AXP_s005
                                 + 0.813325 w_k1_DIS_s005
                                 + 0.965216 w_k1_FDX_s005
                                 + 0.632901 w_k1_IBM_s005
                                 + 0.968503 w_k1_INTC_s005
                                 + 0.698676 w_k1_KO_s005
                                 + 0.618891 w_k1_MAT_s005
                                 + 0.926219 w_k1_MSFT_s005
                                 + 0.762456 w_k1_PG_s005 >= 1
 z_k1_s0056_constraint:          var_1 + z_k1_s0056 + 1.37889 w_k1_ADM_s005
                                 + 1.26276 w_k1_AXP_s005
                                 + 1.46557 w_k1_DIS_s005 + 1.4378 w_k1_FDX_s005
                                 + 1.09756 w_k1_IBM_s005
                                 + 0.984178 w_k1_INTC_s005
                                 + 1.16618 w_k1_KO_s005 + 1.67765 w_k1_MAT_s005
                                 + 1.23975 w_k1_MSFT_s005
                                 + 1.10815 w_k1_PG_s005 >= 1
 z_k1_s0057_constraint:          var_1 + z_k1_s0057 + 1.55049 w_k1_ADM_s005
                                 + 1.6245 w_k1_AXP_s005 + 1.22205 w_k1_DIS_s005
                                 + 1.22198 w_k1_FDX_s005
                                 + 1.48406 w_k1_IBM_s005
                                 + 1.18913 w_k1_INTC_s005
                                 + 1.22897 w_k1_KO_s005 + 1.46406 w_k1_MAT_s005
                                 + 1.14219 w_k1_MSFT_s005
                                 + 1.04686 w_k1_PG_s005 >= 1
 z_k1_s0060_constraint:          var_1 + z_k1_s0060 + 1.2153 w_k1_ADM_s006
                                 + 1.07318 w_k1_AXP_s006
                                 + 0.926219 w_k1_DIS_s006
                                 + 1.06257 w_k1_FDX_s006
                                 + 1.12349 w_k1_IBM_s006
                                 + 1.17931 w_k1_INTC_s006
                                 + 0.985589 w_k1_KO_s006
                                 + 1.01278 w_k1_MAT_s006
                                 + 1.11917 w_k1_MSFT_s006
                                 + 1.01054 w_k1_PG_s006 >= 1
 z_k1_s0061_constraint:          var_1 + z_k1_s0061 + 0.922543 w_k1_ADM_s006
                                 + 0.867986 w_k1_AXP_s006 + 1.029 w_k1_DIS_s006
                                 + 0.920762 w_k1_FDX_s006
                                 + 1.34926 w_k1_IBM_s006
                                 + 1.0131 w_k1_INTC_s006 + 1.11342 w_k1_KO_s006
                                 + 1.6662 w_k1_MAT_s006
                                 + 1.10862 w_k1_MSFT_s006
                                 + 1.20235 w_k1_PG_s006 >= 1
 z_k1_s0062_constraint:          var_1 + z_k1_s0062 + 1.26161 w_k1_ADM_s006
                                 + 1.37351 w_k1_AXP_s006
                                 + 1.71241 w_k1_DIS_s006
                                 + 1.28706 w_k1_FDX_s006 + 1.5074 w_k1_IBM_s006
                                 + 1.62575 w_k1_INTC_s006
                                 + 1.50493 w_k1_KO_s006 + 1.79598 w_k1_MAT_s006
                                 + 1.41271 w_k1_MSFT_s006
                                 + 1.19372 w_k1_PG_s006 >= 1
 z_k1_s0063_constraint:          var_1 + z_k1_s0063 + 1.16067 w_k1_ADM_s006
                                 + 0.887734 w_k1_AXP_s006
                                 + 1.15952 w_k1_DIS_s006
                                 + 1.23549 w_k1_FDX_s006
                                 + 0.811307 w_k1_IBM_s006
                                 + 1.09377 w_k1_INTC_s006
                                 + 0.920681 w_k1_KO_s006
                                 + 0.92288 w_k1_MAT_s006
                                 + 1.10317 w_k1_MSFT_s006
                                 + 1.06533 w_k1_PG_s006 >= 1
 z_k1_s0064_constraint:          var_1 + z_k1_s0064 + 0.759644 w_k1_ADM_s006
                                 + 0.720603 w_k1_AXP_s006
                                 + 0.648278 w_k1_DIS_s006
                                 + 0.778209 w_k1_FDX_s006
                                 + 0.6629 w_k1_IBM_s006
                                 + 0.812145 w_k1_INTC_s006
                                 + 0.737183 w_k1_KO_s006
                                 + 0.864695 w_k1_MAT_s006
                                 + 0.912736 w_k1_MSFT_s006
                                 + 0.853203 w_k1_PG_s006 >= 1
 z_k1_s0065_constraint:          var_1 + z_k1_s0065 + 0.986586 w_k1_ADM_s006
                                 + 0.875171 w_k1_AXP_s006
                                 + 1.16579 w_k1_DIS_s006
                                 + 0.957089 w_k1_FDX_s006
                                 + 0.867805 w_k1_IBM_s006
                                 + 0.991378 w_k1_INTC_s006
                                 + 1.01297 w_k1_KO_s006
                                 + 0.952266 w_k1_MAT_s006
                                 + 1.06611 w_k1_MSFT_s006
                                 + 0.961508 w_k1_PG_s006 >= 1
 z_k1_s0066_constraint:          var_1 + z_k1_s0066 + 1.08628 w_k1_ADM_s006
                                 + 1.09692 w_k1_AXP_s006
                                 + 0.859895 w_k1_DIS_s006
                                 + 1.04313 w_k1_FDX_s006
                                 + 0.668122 w_k1_IBM_s006
                                 + 0.920978 w_k1_INTC_s006
                                 + 0.739282 w_k1_KO_s006
                                 + 0.897979 w_k1_MAT_s006
                                 + 1.00037 w_k1_MSFT_s006
                                 + 0.923337 w_k1_PG_s006 >= 1
 z_k1_s0067_constraint:          var_1 + z_k1_s0067 + 1.28971 w_k1_ADM_s006
                                 + 1.22191 w_k1_AXP_s006
                                 + 1.22548 w_k1_DIS_s006
                                 + 1.21449 w_k1_FDX_s006
                                 + 1.11945 w_k1_IBM_s006
                                 + 1.10927 w_k1_INTC_s006 + 1.1836 w_k1_KO_s006
                                 + 1.74635 w_k1_MAT_s006
                                 + 1.3131 w_k1_MSFT_s006 + 1.1674 w_k1_PG_s006
                                 >= 1
 z_k1_s0070_constraint:          var_1 + z_k1_s0070 + 1.15166 w_k1_ADM_s007
                                 + 0.987697 w_k1_AXP_s007
                                 + 0.892415 w_k1_DIS_s007
                                 + 0.991662 w_k1_FDX_s007
                                 + 0.869332 w_k1_IBM_s007
                                 + 1.06318 w_k1_INTC_s007
                                 + 0.951783 w_k1_KO_s007
                                 + 0.717269 w_k1_MAT_s007
                                 + 1.05716 w_k1_MSFT_s007
                                 + 0.960509 w_k1_PG_s007 >= 1
 z_k1_s0071_constraint:          var_1 + z_k1_s0071 + 1.15178 w_k1_ADM_s007
                                 + 1.39369 w_k1_AXP_s007
                                 + 1.77461 w_k1_DIS_s007
                                 + 1.28274 w_k1_FDX_s007
                                 + 1.59451 w_k1_IBM_s007
                                 + 1.34698 w_k1_INTC_s007
                                 + 1.64252 w_k1_KO_s007 + 1.45842 w_k1_MAT_s007
                                 + 1.36036 w_k1_MSFT_s007
                                 + 1.08227 w_k1_PG_s007 >= 1
 z_k1_s0072_constraint:          var_1 + z_k1_s0072 + 1.14418 w_k1_ADM_s007
                                 + 1.14978 w_k1_AXP_s007
                                 + 1.07169 w_k1_DIS_s007
                                 + 0.980795 w_k1_FDX_s007
                                 + 1.65823 w_k1_IBM_s007
                                 + 0.842207 w_k1_INTC_s007
                                 + 1.26644 w_k1_KO_s007 + 1.2736 w_k1_MAT_s007
                                 + 0.966495 w_k1_MSFT_s007
                                 + 0.996682 w_k1_PG_s007 >= 1
 z_k1_s0073_constraint:          var_1 + z_k1_s0073 + 0.999452 w_k1_ADM_s007
                                 + 1.02338 w_k1_AXP_s007
                                 + 1.24933 w_k1_DIS_s007
                                 + 1.01087 w_k1_FDX_s007
                                 + 1.36603 w_k1_IBM_s007
                                 + 1.1448 w_k1_INTC_s007 + 1.37656 w_k1_KO_s007
                                 + 1.02771 w_k1_MAT_s007
                                 + 1.07911 w_k1_MSFT_s007
                                 + 1.11699 w_k1_PG_s007 >= 1
 z_k1_s0074_constraint:          var_1 + z_k1_s0074 + 0.796616 w_k1_ADM_s007
                                 + 0.667315 w_k1_AXP_s007
                                 + 0.550671 w_k1_DIS_s007
                                 + 0.739767 w_k1_FDX_s007
                                 + 0.7113 w_k1_IBM_s007
                                 + 0.944989 w_k1_INTC_s007
                                 + 0.707164 w_k1_KO_s007
                                 + 0.615319 w_k1_MAT_s007
                                 + 0.808393 w_k1_MSFT_s007
                                 + 0.774173 w_k1_PG_s007 >= 1
 z_k1_s0075_constraint:          var_1 + z_k1_s0075 + 1.16758 w_k1_ADM_s007
                                 + 0.973242 w_k1_AXP_s007
                                 + 1.64372 w_k1_DIS_s007
                                 + 1.23801 w_k1_FDX_s007
                                 + 0.991568 w_k1_IBM_s007
                                 + 1.18798 w_k1_INTC_s007
                                 + 1.26142 w_k1_KO_s007
                                 + 0.705283 w_k1_MAT_s007
                                 + 1.28167 w_k1_MSFT_s007
                                 + 1.08913 w_k1_PG_s007 >= 1
 z_k1_s0076_constraint:          var_1 + z_k1_s0076 + 1.69197 w_k1_ADM_s007
                                 + 1.43291 w_k1_AXP_s007
                                 + 1.37469 w_k1_DIS_s007
                                 + 1.34874 w_k1_FDX_s007
                                 + 0.756681 w_k1_IBM_s007
                                 + 0.91666 w_k1_INTC_s007
                                 + 1.08103 w_k1_KO_s007
                                 + 0.809244 w_k1_MAT_s007
                                 + 1.15901 w_k1_MSFT_s007
                                 + 1.13637 w_k1_PG_s007 >= 1
 z_k1_s0077_constraint:          var_1 + z_k1_s0077 + 0.867376 w_k1_ADM_s007
                                 + 0.760239 w_k1_AXP_s007
                                 + 0.831271 w_k1_DIS_s007
                                 + 0.808859 w_k1_FDX_s007 + 1.242 w_k1_IBM_s007
                                 + 0.935105 w_k1_INTC_s007
                                 + 1.1084 w_k1_KO_s007 + 0.892424 w_k1_MAT_s007
                                 + 0.941451 w_k1_MSFT_s007
                                 + 0.810652 w_k1_PG_s007 >= 1
 z_k1_s0100_constraint:          var_1 + z_k1_s0100 + 0.823969 w_k1_ADM_s010
                                 + 0.755422 w_k1_AXP_s010
                                 + 0.780198 w_k1_DIS_s010
                                 + 0.767161 w_k1_FDX_s010
                                 + 0.983654 w_k1_IBM_s010
                                 + 0.957956 w_k1_INTC_s010
                                 + 1.00767 w_k1_KO_s010
                                 + 0.766645 w_k1_MAT_s010
                                 + 0.769927 w_k1_MSFT_s010
                                 + 0.845408 w_k1_PG_s010 >= 1
 z_k1_s0101_constraint:          var_1 + z_k1_s0101 + 1.02943 w_k1_ADM_s010
                                 + 0.9749 w_k1_AXP_s010
                                 + 0.961927 w_k1_DIS_s010
                                 + 0.898001 w_k1_FDX_s010
                                 + 0.908962 w_k1_IBM_s010
                                 + 0.911996 w_k1_INTC_s010
                                 + 1.16749 w_k1_KO_s010 + 1.05067 w_k1_MAT_s010
                                 + 0.976745 w_k1_MSFT_s010
                                 + 1.15128 w_k1_PG_s010 >= 1
 z_k1_s0102_constraint:          var_1 + z_k1_s0102 + 0.999104 w_k1_ADM_s010
                                 + 1.00422 w_k1_AXP_s010
                                 + 1.22122 w_k1_DIS_s010
                                 + 1.12854 w_k1_FDX_s010
                                 + 0.791037 w_k1_IBM_s010
                                 + 1.11239 w_k1_INTC_s010
                                 + 1.18588 w_k1_KO_s010
                                 + 0.859147 w_k1_MAT_s010
                                 + 1.13516 w_k1_MSFT_s010
                                 + 0.987898 w_k1_PG_s010 >= 1
 z_k1_s0103_constraint:          var_1 + z_k1_s0103 + 0.946882 w_k1_ADM_s010
                                 + 0.946881 w_k1_AXP_s010
                                 + 0.660086 w_k1_DIS_s010
                                 + 0.88676 w_k1_FDX_s010
                                 + 0.705812 w_k1_IBM_s010
                                 + 0.950579 w_k1_INTC_s010
                                 + 0.815402 w_k1_KO_s010
                                 + 0.790394 w_k1_MAT_s010
                                 + 0.84881 w_k1_MSFT_s010
                                 + 0.878329 w_k1_PG_s010 >= 1
 z_k1_s0104_constraint:          var_1 + z_k1_s0104 + 0.831679 w_k1_ADM_s010
                                 + 0.919279 w_k1_AXP_s010
                                 + 0.912646 w_k1_DIS_s010
                                 + 0.81316 w_k1_FDX_s010
                                 + 1.43469 w_k1_IBM_s010
                                 + 0.987696 w_k1_INTC_s010
                                 + 1.34277 w_k1_KO_s010 + 1.59613 w_k1_MAT_s010
                                 + 0.84468 w_k1_MSFT_s010
                                 + 1.05605 w_k1_PG_s010 >= 1
 z_k1_s0105_constraint:          var_1 + z_k1_s0105 + 1.33896 w_k1_ADM_s010
                                 + 1.31534 w_k1_AXP_s010
                                 + 1.21458 w_k1_DIS_s010
                                 + 1.23597 w_k1_FDX_s010
                                 + 1.30231 w_k1_IBM_s010
                                 + 1.16292 w_k1_INTC_s010
                                 + 1.26406 w_k1_KO_s010 + 1.77885 w_k1_MAT_s010
                                 + 1.04923 w_k1_MSFT_s010 + 1.1457 w_k1_PG_s010
                                 >= 1
 z_k1_s0106_constraint:          var_1 + z_k1_s0106 + 0.976555 w_k1_ADM_s010
                                 + 0.832756 w_k1_AXP_s010
                                 + 0.897023 w_k1_DIS_s010
                                 + 0.909758 w_k1_FDX_s010
                                 + 1.08116 w_k1_IBM_s010
                                 + 0.941404 w_k1_INTC_s010
                                 + 1.16429 w_k1_KO_s010 + 1.02948 w_k1_MAT_s010
                                 + 0.872315 w_k1_MSFT_s010
                                 + 1.02867 w_k1_PG_s010 >= 1
 z_k1_s0107_constraint:          var_1 + z_k1_s0107 + 0.89579 w_k1_ADM_s010
                                 + 0.655654 w_k1_AXP_s010
                                 + 0.610492 w_k1_DIS_s010
                                 + 0.721977 w_k1_FDX_s010
                                 + 0.684365 w_k1_IBM_s010
                                 + 0.820757 w_k1_INTC_s010
                                 + 0.761216 w_k1_KO_s010
                                 + 0.470613 w_k1_MAT_s010
                                 + 0.68988 w_k1_MSFT_s010
                                 + 0.806168 w_k1_PG_s010 >= 1
 z_k1_s0110_constraint:          var_1 + z_k1_s0110 + 1.12666 w_k1_ADM_s011
                                 + 0.858991 w_k1_AXP_s011
                                 + 0.786572 w_k1_DIS_s011
                                 + 0.944378 w_k1_FDX_s011
                                 + 1.16037 w_k1_IBM_s011
                                 + 0.94108 w_k1_INTC_s011
                                 + 0.959311 w_k1_KO_s011
                                 + 1.24014 w_k1_MAT_s011
                                 + 0.966702 w_k1_MSFT_s011
                                 + 1.02972 w_k1_PG_s011 >= 1
 z_k1_s0111_constraint:          var_1 + z_k1_s0111 + 0.951479 w_k1_ADM_s011
                                 + 0.819527 w_k1_AXP_s011
                                 + 0.814819 w_k1_DIS_s011
                                 + 0.84702 w_k1_FDX_s011 + 1.0178 w_k1_IBM_s011
                                 + 1.01968 w_k1_INTC_s011
                                 + 1.00931 w_k1_KO_s011
                                 + 0.987359 w_k1_MAT_s011
                                 + 0.875143 w_k1_MSFT_s011
                                 + 0.979057 w_k1_PG_s011 >= 1
 z_k1_s0112_constraint:          var_1 + z_k1_s0112 + 0.918129 w_k1_ADM_s011
                                 + 0.675677 w_k1_AXP_s011
                                 + 0.677511 w_k1_DIS_s011
                                 + 0.832861 w_k1_FDX_s011
                                 + 0.742554 w_k1_IBM_s011
                                 + 0.844472 w_k1_INTC_s011
                                 + 0.702956 w_k1_KO_s011
                                 + 0.69725 w_k1_MAT_s011
                                 + 0.815424 w_k1_MSFT_s011
                                 + 0.851499 w_k1_PG_s011 >= 1
 z_k1_s0113_constraint:          var_1 + z_k1_s0113 + 1.04087 w_k1_ADM_s011
                                 + 1.11332 w_k1_AXP_s011
                                 + 1.10587 w_k1_DIS_s011
                                 + 1.16086 w_k1_FDX_s011
                                 + 1.02236 w_k1_IBM_s011
                                 + 1.16555 w_k1_INTC_s011
                                 + 1.07994 w_k1_KO_s011 + 1.01171 w_k1_MAT_s011
                                 + 1.07534 w_k1_MSFT_s011
                                 + 1.06254 w_k1_PG_s011 >= 1
 z_k1_s0114_constraint:          var_1 + z_k1_s0114 + 1.18131 w_k1_ADM_s011
                                 + 1.17326 w_k1_AXP_s011
                                 + 0.967905 w_k1_DIS_s011
                                 + 1.0317 w_k1_FDX_s011
                                 + 0.901714 w_k1_IBM_s011
                                 + 1.16715 w_k1_INTC_s011
                                 + 0.957245 w_k1_KO_s011
                                 + 0.792075 w_k1_MAT_s011
                                 + 1.0488 w_k1_MSFT_s011 + 1.03139 w_k1_PG_s011
                                 >= 1
 z_k1_s0115_constraint:          var_1 + z_k1_s0115 + 1.41572 w_k1_ADM_s011
                                 + 1.41819 w_k1_AXP_s011
                                 + 1.62738 w_k1_DIS_s011 + 1.4402 w_k1_FDX_s011
                                 + 1.28904 w_k1_IBM_s011
                                 + 1.14297 w_k1_INTC_s011
                                 + 1.24156 w_k1_KO_s011
                                 + 0.920547 w_k1_MAT_s011
                                 + 1.16734 w_k1_MSFT_s011
                                 + 1.35013 w_k1_PG_s011 >= 1
 z_k1_s0116_constraint:          var_1 + z_k1_s0116 + 0.964406 w_k1_ADM_s011
                                 + 0.941005 w_k1_AXP_s011
                                 + 1.06477 w_k1_DIS_s011
                                 + 1.06067 w_k1_FDX_s011
                                 + 1.32667 w_k1_IBM_s011
                                 + 1.13669 w_k1_INTC_s011
                                 + 1.18664 w_k1_KO_s011 + 1.51288 w_k1_MAT_s011
                                 + 1.01774 w_k1_MSFT_s011
                                 + 1.12967 w_k1_PG_s011 >= 1
 z_k1_s0117_constraint:          var_1 + z_k1_s0117 + 0.959259 w_k1_ADM_s011
                                 + 0.820249 w_k1_AXP_s011
                                 + 1.12573 w_k1_DIS_s011
                                 + 1.16659 w_k1_FDX_s011
                                 + 0.765289 w_k1_IBM_s011
                                 + 1.39844 w_k1_INTC_s011
                                 + 0.978746 w_k1_KO_s011
                                 + 0.53738 w_k1_MAT_s011
                                 + 1.15541 w_k1_MSFT_s011
                                 + 1.07987 w_k1_PG_s011 >= 1
 z_k1_s0120_constraint:          var_1 + z_k1_s0120 + 0.905344 w_k1_ADM_s012
                                 + 1.29717 w_k1_AXP_s012
                                 + 1.00906 w_k1_DIS_s012
                                 + 0.945158 w_k1_FDX_s012
                                 + 0.967747 w_k1_IBM_s012
                                 + 0.984824 w_k1_INTC_s012
                                 + 0.749937 w_k1_KO_s012
                                 + 1.06611 w_k1_MAT_s012
                                 + 0.841923 w_k1_MSFT_s012
                                 + 1.03642 w_k1_PG_s012 >= 1
 z_k1_s0121_constraint:          var_1 + z_k1_s0121 + 1.28506 w_k1_ADM_s012
                                 + 1.4161 w_k1_AXP_s012 + 1.66557 w_k1_DIS_s012
                                 + 1.40625 w_k1_FDX_s012 + 1.1234 w_k1_IBM_s012
                                 + 1.38795 w_k1_INTC_s012
                                 + 1.05672 w_k1_KO_s012 + 1.04945 w_k1_MAT_s012
                                 + 1.194 w_k1_MSFT_s012 + 1.24907 w_k1_PG_s012
                                 >= 1
 z_k1_s0122_constraint:          var_1 + z_k1_s0122 + 0.830477 w_k1_ADM_s012
                                 + 0.690932 w_k1_AXP_s012
                                 + 0.749336 w_k1_DIS_s012
                                 + 0.87575 w_k1_FDX_s012
                                 + 0.68991 w_k1_IBM_s012
                                 + 0.940607 w_k1_INTC_s012
                                 + 0.718454 w_k1_KO_s012
                                 + 0.840669 w_k1_MAT_s012
                                 + 0.858696 w_k1_MSFT_s012
                                 + 0.882962 w_k1_PG_s012 >= 1
 z_k1_s0123_constraint:          var_1 + z_k1_s0123 + 1.01811 w_k1_ADM_s012
                                 + 1.04959 w_k1_AXP_s012
                                 + 1.31985 w_k1_DIS_s012 + 1.1687 w_k1_FDX_s012
                                 + 1.14532 w_k1_IBM_s012
                                 + 1.37974 w_k1_INTC_s012 + 1.3035 w_k1_KO_s012
                                 + 0.983417 w_k1_MAT_s012
                                 + 1.19235 w_k1_MSFT_s012
                                 + 1.17643 w_k1_PG_s012 >= 1
 z_k1_s0124_constraint:          var_1 + z_k1_s0124 + 0.860934 w_k1_ADM_s012
                                 + 1.2295 w_k1_AXP_s012 + 1.15953 w_k1_DIS_s012
                                 + 0.993623 w_k1_FDX_s012
                                 + 1.47291 w_k1_IBM_s012
                                 + 1.1859 w_k1_INTC_s012 + 1.1883 w_k1_KO_s012
                                 + 1.48291 w_k1_MAT_s012
                                 + 0.834198 w_k1_MSFT_s012
                                 + 1.02505 w_k1_PG_s012 >= 1
 z_k1_s0125_constraint:          var_1 + z_k1_s0125 + 0.734354 w_k1_ADM_s012
                                 + 0.764987 w_k1_AXP_s012
                                 + 0.750359 w_k1_DIS_s012
                                 + 0.806749 w_k1_FDX_s012
                                 + 1.03189 w_k1_IBM_s012
                                 + 1.05777 w_k1_INTC_s012
                                 + 0.881384 w_k1_KO_s012
                                 + 0.762205 w_k1_MAT_s012
                                 + 0.818722 w_k1_MSFT_s012
                                 + 0.92216 w_k1_PG_s012 >= 1
 z_k1_s0126_constraint:          var_1 + z_k1_s0126 + 1.13493 w_k1_ADM_s012
                                 + 1.03944 w_k1_AXP_s012
                                 + 1.07917 w_k1_DIS_s012
                                 + 1.20363 w_k1_FDX_s012
                                 + 0.724514 w_k1_IBM_s012
                                 + 1.05871 w_k1_INTC_s012
                                 + 0.809895 w_k1_KO_s012
                                 + 0.748454 w_k1_MAT_s012
                                 + 0.978564 w_k1_MSFT_s012
                                 + 1.03424 w_k1_PG_s012 >= 1
 z_k1_s0127_constraint:          var_1 + z_k1_s0127 + 0.997534 w_k1_ADM_s012
                                 + 1.27449 w_k1_AXP_s012 + 1.4468 w_k1_DIS_s012
                                 + 1.23496 w_k1_FDX_s012
                                 + 1.53288 w_k1_IBM_s012
                                 + 1.28926 w_k1_INTC_s012 + 1.1919 w_k1_KO_s012
                                 + 1.29363 w_k1_MAT_s012
                                 + 1.01937 w_k1_MSFT_s012
                                 + 1.22625 w_k1_PG_s012 >= 1
 z_k1_s0130_constraint:          var_1 + z_k1_s0130 + 0.860192 w_k1_ADM_s013
                                 + 0.791972 w_k1_AXP_s013
                                 + 0.568908 w_k1_DIS_s013
                                 + 0.752425 w_k1_FDX_s013
                                 + 0.916337 w_k1_IBM_s013
                                 + 0.763888 w_k1_INTC_s013
                                 + 0.798671 w_k1_KO_s013
                                 + 0.854559 w_k1_MAT_s013
                                 + 0.833804 w_k1_MSFT_s013
                                 + 0.875876 w_k1_PG_s013 >= 1
 z_k1_s0131_constraint:          var_1 + z_k1_s0131 + 1.34788 w_k1_ADM_s013
                                 + 1.23896 w_k1_AXP_s013
                                 + 1.07343 w_k1_DIS_s013
                                 + 1.05512 w_k1_FDX_s013 + 1.1162 w_k1_IBM_s013
                                 + 0.940697 w_k1_INTC_s013
                                 + 1.06397 w_k1_KO_s013 + 1.2756 w_k1_MAT_s013
                                 + 1.22228 w_k1_MSFT_s013
                                 + 1.01367 w_k1_PG_s013 >= 1
 z_k1_s0132_constraint:          var_1 + z_k1_s0132 + 0.859197 w_k1_ADM_s013
                                 + 0.883174 w_k1_AXP_s013
                                 + 0.674896 w_k1_DIS_s013
                                 + 0.898509 w_k1_FDX_s013
                                 + 0.627847 w_k1_IBM_s013
                                 + 0.688008 w_k1_INTC_s013
                                 + 0.710275 w_k1_KO_s013
                                 + 0.680495 w_k1_MAT_s013
                                 + 0.880817 w_k1_MSFT_s013
                                 + 0.832707 w_k1_PG_s013 >= 1
 z_k1_s0133_constraint:          var_1 + z_k1_s0133 + 1.2846 w_k1_ADM_s013
                                 + 0.798323 w_k1_AXP_s013
                                 + 0.91604 w_k1_DIS_s013
                                 + 0.937218 w_k1_FDX_s013
                                 + 1.33895 w_k1_IBM_s013
                                 + 0.714396 w_k1_INTC_s013
                                 + 1.14814 w_k1_KO_s013
                                 + 0.981661 w_k1_MAT_s013
                                 + 0.98242 w_k1_MSFT_s013
                                 + 0.935054 w_k1_PG_s013 >= 1
 z_k1_s0134_constraint:          var_1 + z_k1_s0134 + 1.15422 w_k1_ADM_s013
                                 + 0.980097 w_k1_AXP_s013
                                 + 0.932255 w_k1_DIS_s013
                                 + 0.967917 w_k1_FDX_s013
                                 + 0.794761 w_k1_IBM_s013
                                 + 0.831147 w_k1_INTC_s013
                                 + 0.926953 w_k1_KO_s013
                                 + 0.664918 w_k1_MAT_s013
                                 + 1.02142 w_k1_MSFT_s013
                                 + 0.837279 w_k1_PG_s013 >= 1
 z_k1_s0135_constraint:          var_1 + z_k1_s0135 + 1.53594 w_k1_ADM_s013
                                 + 1.74993 w_k1_AXP_s013 + 1.7299 w_k1_DIS_s013
                                 + 1.4672 w_k1_FDX_s013 + 1.88233 w_k1_IBM_s013
                                 + 1.30719 w_k1_INTC_s013
                                 + 1.68701 w_k1_KO_s013 + 1.39058 w_k1_MAT_s013
                                 + 1.43969 w_k1_MSFT_s013
                                 + 1.32986 w_k1_PG_s013 >= 1
 z_k1_s0136_constraint:          var_1 + z_k1_s0136 + 1.26926 w_k1_ADM_s013
                                 + 1.16137 w_k1_AXP_s013
                                 + 1.38465 w_k1_DIS_s013
                                 + 1.27193 w_k1_FDX_s013
                                 + 0.915517 w_k1_IBM_s013
                                 + 1.17116 w_k1_INTC_s013
                                 + 1.23817 w_k1_KO_s013 + 1.09643 w_k1_MAT_s013
                                 + 1.36461 w_k1_MSFT_s013
                                 + 1.17641 w_k1_PG_s013 >= 1
 z_k1_s0137_constraint:          var_1 + z_k1_s0137 + 1.24012 w_k1_ADM_s013
                                 + 1.31221 w_k1_AXP_s013 + 1.0404 w_k1_DIS_s013
                                 + 1.01779 w_k1_FDX_s013
                                 + 1.53242 w_k1_IBM_s013
                                 + 1.08578 w_k1_INTC_s013
                                 + 1.30992 w_k1_KO_s013 + 1.50207 w_k1_MAT_s013
                                 + 1.00842 w_k1_MSFT_s013
                                 + 0.986191 w_k1_PG_s013 >= 1
 z_k1_s0140_constraint:          var_1 + z_k1_s0140 + 1.11282 w_k1_ADM_s014
                                 + 1.20909 w_k1_AXP_s014
                                 + 1.28475 w_k1_DIS_s014
                                 + 1.13757 w_k1_FDX_s014
                                 + 1.12814 w_k1_IBM_s014
                                 + 1.35651 w_k1_INTC_s014
                                 + 1.19495 w_k1_KO_s014 + 1.2001 w_k1_MAT_s014
                                 + 1.23051 w_k1_MSFT_s014
                                 + 1.22504 w_k1_PG_s014 >= 1
 z_k1_s0141_constraint:          var_1 + z_k1_s0141 + 1.18441 w_k1_ADM_s014
                                 + 1.6675 w_k1_AXP_s014 + 1.62982 w_k1_DIS_s014
                                 + 1.44811 w_k1_FDX_s014
                                 + 1.01804 w_k1_IBM_s014
                                 + 1.51846 w_k1_INTC_s014
                                 + 1.28599 w_k1_KO_s014 + 1.77998 w_k1_MAT_s014
                                 + 1.41347 w_k1_MSFT_s014
                                 + 1.25148 w_k1_PG_s014 >= 1
 z_k1_s0142_constraint:          var_1 + z_k1_s0142 + 0.842794 w_k1_ADM_s014
                                 + 0.88833 w_k1_AXP_s014
                                 + 0.811321 w_k1_DIS_s014
                                 + 0.809588 w_k1_FDX_s014
                                 + 1.13142 w_k1_IBM_s014
                                 + 1.29609 w_k1_INTC_s014
                                 + 0.93564 w_k1_KO_s014
                                 + 0.905841 w_k1_MAT_s014
                                 + 0.823641 w_k1_MSFT_s014
                                 + 0.925492 w_k1_PG_s014 >= 1
 z_k1_s0143_constraint:          var_1 + z_k1_s0143 + 1.14913 w_k1_ADM_s014
                                 + 1.06606 w_k1_AXP_s014 + 1.0014 w_k1_DIS_s014
                                 + 1.07079 w_k1_FDX_s014
                                 + 0.695494 w_k1_IBM_s014
                                 + 1.36973 w_k1_INTC_s014
                                 + 0.794548 w_k1_KO_s014
                                 + 0.734902 w_k1_MAT_s014
                                 + 1.06449 w_k1_MSFT_s014
                                 + 0.924535 w_k1_PG_s014 >= 1
 z_k1_s0144_constraint:          var_1 + z_k1_s0144 + 1.12169 w_k1_ADM_s014
                                 + 0.8983 w_k1_AXP_s014
                                 + 0.767439 w_k1_DIS_s014
                                 + 0.949553 w_k1_FDX_s014
                                 + 0.789762 w_k1_IBM_s014
                                 + 1.21521 w_k1_INTC_s014
                                 + 0.809254 w_k1_KO_s014
                                 + 1.04298 w_k1_MAT_s014
                                 + 0.88548 w_k1_MSFT_s014
                                 + 1.00666 w_k1_PG_s014 >= 1
 z_k1_s0145_constraint:          var_1 + z_k1_s0145 + 0.870584 w_k1_ADM_s014
                                 + 1.19483 w_k1_AXP_s014
                                 + 1.04305 w_k1_DIS_s014
                                 + 0.987458 w_k1_FDX_s014
                                 + 1.20313 w_k1_IBM_s014
                                 + 1.33031 w_k1_INTC_s014
                                 + 0.962516 w_k1_KO_s014
                                 + 0.978263 w_k1_MAT_s014
                                 + 0.920563 w_k1_MSFT_s014
                                 + 1.00032 w_k1_PG_s014 >= 1
 z_k1_s0146_constraint:          var_1 + z_k1_s0146 + 0.777847 w_k1_ADM_s014
                                 + 0.65676 w_k1_AXP_s014
                                 + 0.559838 w_k1_DIS_s014
                                 + 0.64839 w_k1_FDX_s014
                                 + 1.08754 w_k1_IBM_s014
                                 + 0.91031 w_k1_INTC_s014
                                 + 0.769028 w_k1_KO_s014
                                 + 0.692206 w_k1_MAT_s014
                                 + 0.680838 w_k1_MSFT_s014
                                 + 0.776752 w_k1_PG_s014 >= 1
 z_k1_s0147_constraint:          var_1 + z_k1_s0147 + 1.03078 w_k1_ADM_s014
                                 + 1.39444 w_k1_AXP_s014
                                 + 1.00472 w_k1_DIS_s014
                                 + 1.16615 w_k1_FDX_s014
                                 + 1.22493 w_k1_IBM_s014
                                 + 1.36412 w_k1_INTC_s014
                                 + 0.970215 w_k1_KO_s014
                                 + 1.34349 w_k1_MAT_s014
                                 + 1.03222 w_k1_MSFT_s014
                                 + 1.10691 w_k1_PG_s014 >= 1
 z_k1_s0150_constraint:          var_1 + z_k1_s0150 + 1.18552 w_k1_ADM_s015
                                 + 1.06897 w_k1_AXP_s015
                                 + 1.07673 w_k1_DIS_s015
                                 + 1.12015 w_k1_FDX_s015
                                 + 0.70093 w_k1_IBM_s015
                                 + 1.15765 w_k1_INTC_s015
                                 + 0.983713 w_k1_KO_s015
                                 + 0.748443 w_k1_MAT_s015
                                 + 1.13571 w_k1_MSFT_s015
                                 + 0.853137 w_k1_PG_s015 >= 1
 z_k1_s0151_constraint:          var_1 + z_k1_s0151 + 1.01555 w_k1_ADM_s015
                                 + 1.3998 w_k1_AXP_s015
                                 + 0.962675 w_k1_DIS_s015
                                 + 1.14358 w_k1_FDX_s015
                                 + 1.08353 w_k1_IBM_s015
                                 + 1.08753 w_k1_INTC_s015
                                 + 0.958352 w_k1_KO_s015
                                 + 1.34758 w_k1_MAT_s015
                                 + 0.973334 w_k1_MSFT_s015
                                 + 1.06854 w_k1_PG_s015 >= 1
 z_k1_s0152_constraint:          var_1 + z_k1_s0152 + 1.3402 w_k1_ADM_s015
                                 + 1.37082 w_k1_AXP_s015
                                 + 1.41021 w_k1_DIS_s015
                                 + 1.39696 w_k1_FDX_s015
                                 + 0.861396 w_k1_IBM_s015
                                 + 1.47772 w_k1_INTC_s015
                                 + 1.09033 w_k1_KO_s015
                                 + 0.662839 w_k1_MAT_s015
                                 + 1.24814 w_k1_MSFT_s015
                                 + 1.10846 w_k1_PG_s015 >= 1
 z_k1_s0153_constraint:          var_1 + z_k1_s0153 + 0.862492 w_k1_ADM_s015
                                 + 1.26352 w_k1_AXP_s015
                                 + 0.852425 w_k1_DIS_s015
                                 + 0.958199 w_k1_FDX_s015
                                 + 1.59948 w_k1_IBM_s015
                                 + 0.950885 w_k1_INTC_s015
                                 + 1.36677 w_k1_KO_s015 + 1.7693 w_k1_MAT_s015
                                 + 0.964705 w_k1_MSFT_s015
                                 + 1.08003 w_k1_PG_s015 >= 1
 z_k1_s0154_constraint:          var_1 + z_k1_s0154 + 0.873617 w_k1_ADM_s015
                                 + 0.850852 w_k1_AXP_s015
                                 + 1.16314 w_k1_DIS_s015
                                 + 0.937018 w_k1_FDX_s015
                                 + 1.06994 w_k1_IBM_s015
                                 + 1.10933 w_k1_INTC_s015
                                 + 1.17463 w_k1_KO_s015
                                 + 0.918724 w_k1_MAT_s015
                                 + 1.11461 w_k1_MSFT_s015
                                 + 0.996426 w_k1_PG_s015 >= 1
 z_k1_s0155_constraint:          var_1 + z_k1_s0155 + 0.770017 w_k1_ADM_s015
                                 + 0.796632 w_k1_AXP_s015
                                 + 0.756853 w_k1_DIS_s015
                                 + 0.744448 w_k1_FDX_s015
                                 + 0.832292 w_k1_IBM_s015
                                 + 0.852293 w_k1_INTC_s015
                                 + 0.865872 w_k1_KO_s015
                                 + 0.947532 w_k1_MAT_s015
                                 + 0.862067 w_k1_MSFT_s015
                                 + 0.813875 w_k1_PG_s015 >= 1
 z_k1_s0156_constraint:          var_1 + z_k1_s0156 + 0.713827 w_k1_ADM_s015
                                 + 1.01616 w_k1_AXP_s015
                                 + 0.767635 w_k1_DIS_s015
                                 + 0.885368 w_k1_FDX_s015
                                 + 0.803493 w_k1_IBM_s015
                                 + 0.954238 w_k1_INTC_s015
                                 + 0.810023 w_k1_KO_s015
                                 + 1.18789 w_k1_MAT_s015
                                 + 0.767406 w_k1_MSFT_s015
                                 + 0.778968 w_k1_PG_s015 >= 1
 z_k1_s0157_constraint:          var_1 + z_k1_s0157 + 1.09116 w_k1_ADM_s015
                                 + 1.36921 w_k1_AXP_s015
                                 + 1.20669 w_k1_DIS_s015
                                 + 1.18844 w_k1_FDX_s015
                                 + 1.37424 w_k1_IBM_s015
                                 + 1.20558 w_k1_INTC_s015
                                 + 1.21598 w_k1_KO_s015 + 1.42175 w_k1_MAT_s015
                                 + 1.00978 w_k1_MSFT_s015
                                 + 0.996457 w_k1_PG_s015 >= 1
 z_k1_s0160_constraint:          var_1 + z_k1_s0160 + 1.02642 w_k1_ADM_s016
                                 + 1.12281 w_k1_AXP_s016
                                 + 1.08437 w_k1_DIS_s016 + 1.1546 w_k1_FDX_s016
                                 + 0.943195 w_k1_IBM_s016
                                 + 0.857642 w_k1_INTC_s016
                                 + 1.01928 w_k1_KO_s016 + 1.39689 w_k1_MAT_s016
                                 + 1.16187 w_k1_MSFT_s016
                                 + 1.05178 w_k1_PG_s016 >= 1
 z_k1_s0161_constraint:          var_1 + z_k1_s0161 + 0.836818 w_k1_ADM_s016
                                 + 1.01735 w_k1_AXP_s016
                                 + 0.872057 w_k1_DIS_s016
                                 + 0.931354 w_k1_FDX_s016
                                 + 1.14186 w_k1_IBM_s016
                                 + 0.753449 w_k1_INTC_s016
                                 + 0.938946 w_k1_KO_s016
                                 + 1.48574 w_k1_MAT_s016
                                 + 0.915286 w_k1_MSFT_s016
                                 + 0.926672 w_k1_PG_s016 >= 1
 z_k1_s0162_constraint:          var_1 + z_k1_s0162 + 0.905564 w_k1_ADM_s016
                                 + 0.919637 w_k1_AXP_s016
                                 + 1.01756 w_k1_DIS_s016
                                 + 0.893343 w_k1_FDX_s016
                                 + 1.77883 w_k1_IBM_s016
                                 + 0.693732 w_k1_INTC_s016
                                 + 1.39936 w_k1_KO_s016 + 2.58132 w_k1_MAT_s016
                                 + 1.04963 w_k1_MSFT_s016
                                 + 1.05573 w_k1_PG_s016 >= 1
 z_k1_s0163_constraint:          var_1 + z_k1_s0163 + 0.972518 w_k1_ADM_s016
                                 + 0.793506 w_k1_AXP_s016
                                 + 0.695266 w_k1_DIS_s016
                                 + 0.830609 w_k1_FDX_s016
                                 + 0.576257 w_k1_IBM_s016
                                 + 0.704849 w_k1_INTC_s016
                                 + 0.71258 w_k1_KO_s016
                                 + 0.652714 w_k1_MAT_s016
                                 + 0.766043 w_k1_MSFT_s016
                                 + 0.820152 w_k1_PG_s016 >= 1
 z_k1_s0164_constraint:          var_1 + z_k1_s0164 + 0.802979 w_k1_ADM_s016
                                 + 0.782954 w_k1_AXP_s016
                                 + 0.85822 w_k1_DIS_s016
                                 + 0.867662 w_k1_FDX_s016
                                 + 0.722669 w_k1_IBM_s016
                                 + 0.865972 w_k1_INTC_s016
                                 + 0.742594 w_k1_KO_s016
                                 + 0.835253 w_k1_MAT_s016
                                 + 0.940915 w_k1_MSFT_s016
                                 + 0.841732 w_k1_PG_s016 >= 1
 z_k1_s0165_constraint:          var_1 + z_k1_s0165 + 0.828471 w_k1_ADM_s016
                                 + 0.819469 w_k1_AXP_s016
                                 + 0.927268 w_k1_DIS_s016
                                 + 0.872961 w_k1_FDX_s016
                                 + 0.905665 w_k1_IBM_s016
                                 + 0.762167 w_k1_INTC_s016
                                 + 0.813523 w_k1_KO_s016
                                 + 0.983289 w_k1_MAT_s016
                                 + 0.898974 w_k1_MSFT_s016
                                 + 0.838358 w_k1_PG_s016 >= 1
 z_k1_s0166_constraint:          var_1 + z_k1_s0166 + 1.54886 w_k1_ADM_s016
                                 + 1.40351 w_k1_AXP_s016
                                 + 1.45655 w_k1_DIS_s016
                                 + 1.41837 w_k1_FDX_s016
                                 + 0.859485 w_k1_IBM_s016
                                 + 0.919944 w_k1_INTC_s016
                                 + 1.03386 w_k1_KO_s016 + 1.26414 w_k1_MAT_s016
                                 + 1.32887 w_k1_MSFT_s016
                                 + 1.08853 w_k1_PG_s016 >= 1
 z_k1_s0167_constraint:          var_1 + z_k1_s0167 + 1.02379 w_k1_ADM_s016
                                 + 1.14002 w_k1_AXP_s016
                                 + 1.68588 w_k1_DIS_s016
                                 + 1.23285 w_k1_FDX_s016
                                 + 1.45946 w_k1_IBM_s016
                                 + 0.996006 w_k1_INTC_s016
                                 + 1.57589 w_k1_KO_s016 + 1.67866 w_k1_MAT_s016
                                 + 1.38672 w_k1_MSFT_s016 + 1.2157 w_k1_PG_s016
                                 >= 1
 z_k1_s0170_constraint:          var_1 + z_k1_s0170 + 0.751835 w_k1_ADM_s017
                                 + 0.7849 w_k1_AXP_s017
                                 + 0.566818 w_k1_DIS_s017
                                 + 0.668156 w_k1_FDX_s017
                                 + 0.778271 w_k1_IBM_s017
                                 + 0.711505 w_k1_INTC_s017
                                 + 0.766908 w_k1_KO_s017
                                 + 0.758298 w_k1_MAT_s017
                                 + 0.866136 w_k1_MSFT_s017
                                 + 0.843277 w_k1_PG_s017 >= 1
 z_k1_s0171_constraint:          var_1 + z_k1_s0171 + 0.832528 w_k1_ADM_s017
                                 + 0.9509 w_k1_AXP_s017
                                 + 0.743363 w_k1_DIS_s017
                                 + 0.897621 w_k1_FDX_s017
                                 + 0.821159 w_k1_IBM_s017
                                 + 0.862651 w_k1_INTC_s017
                                 + 0.861782 w_k1_KO_s017
                                 + 0.726327 w_k1_MAT_s017
                                 + 0.936718 w_k1_MSFT_s017
                                 + 0.910182 w_k1_PG_s017 >= 1
 z_k1_s0172_constraint:          var_1 + z_k1_s0172 + 0.814077 w_k1_ADM_s017
                                 + 0.889039 w_k1_AXP_s017
                                 + 0.831427 w_k1_DIS_s017
                                 + 0.762503 w_k1_FDX_s017
                                 + 1.07609 w_k1_IBM_s017
                                 + 0.985604 w_k1_INTC_s017
                                 + 0.96873 w_k1_KO_s017
                                 + 0.947823 w_k1_MAT_s017
                                 + 1.00861 w_k1_MSFT_s017
                                 + 0.993885 w_k1_PG_s017 >= 1
 z_k1_s0173_constraint:          var_1 + z_k1_s0173 + 1.1876 w_k1_ADM_s017
                                 + 1.38911 w_k1_AXP_s017
                                 + 1.20338 w_k1_DIS_s017 + 1.2766 w_k1_FDX_s017
                                 + 0.903766 w_k1_IBM_s017
                                 + 1.26073 w_k1_INTC_s017
                                 + 1.19025 w_k1_KO_s017 + 1.0158 w_k1_MAT_s017
                                 + 1.24508 w_k1_MSFT_s017
                                 + 1.17733 w_k1_PG_s017 >= 1
 z_k1_s0174_constraint:          var_1 + z_k1_s0174 + 0.984834 w_k1_ADM_s017
                                 + 1.28119 w_k1_AXP_s017
                                 + 1.24614 w_k1_DIS_s017
                                 + 1.09247 w_k1_FDX_s017
                                 + 1.24832 w_k1_IBM_s017
                                 + 1.30054 w_k1_INTC_s017
                                 + 1.24883 w_k1_KO_s017
                                 + 0.940329 w_k1_MAT_s017
                                 + 1.29793 w_k1_MSFT_s017
                                 + 1.11881 w_k1_PG_s017 >= 1
 z_k1_s0175_constraint:          var_1 + z_k1_s0175 + 0.886788 w_k1_ADM_s017
                                 + 0.711173 w_k1_AXP_s017
                                 + 0.79132 w_k1_DIS_s017
                                 + 0.850913 w_k1_FDX_s017
                                 + 0.61561 w_k1_IBM_s017
                                 + 0.80665 w_k1_INTC_s017
                                 + 0.811876 w_k1_KO_s017
                                 + 0.390172 w_k1_MAT_s017
                                 + 0.961279 w_k1_MSFT_s017
                                 + 0.939331 w_k1_PG_s017 >= 1
 z_k1_s0176_constraint:          var_1 + z_k1_s0176 + 1.06533 w_k1_ADM_s017
                                 + 1.64844 w_k1_AXP_s017
                                 + 1.53478 w_k1_DIS_s017
                                 + 1.08912 w_k1_FDX_s017
                                 + 1.91366 w_k1_IBM_s017
                                 + 1.26286 w_k1_INTC_s017
                                 + 1.81227 w_k1_KO_s017 + 2.72049 w_k1_MAT_s017
                                 + 1.26981 w_k1_MSFT_s017 + 1.3349 w_k1_PG_s017
                                 >= 1
 z_k1_s0177_constraint:          var_1 + z_k1_s0177 + 0.911367 w_k1_ADM_s017
                                 + 1.06324 w_k1_AXP_s017
                                 + 1.00325 w_k1_DIS_s017
                                 + 0.910602 w_k1_FDX_s017
                                 + 0.943814 w_k1_IBM_s017
                                 + 1.1409 w_k1_INTC_s017 + 1.05114 w_k1_KO_s017
                                 + 0.809036 w_k1_MAT_s017
                                 + 1.10395 w_k1_MSFT_s017
                                 + 1.01158 w_k1_PG_s017 >= 1
 z_k1_s1000_constraint:          var_1 + z_k1_s1000 + 0.973784 w_k1_ADM_s100
                                 + 0.881237 w_k1_AXP_s100
                                 + 0.694726 w_k1_DIS_s100
                                 + 0.936549 w_k1_FDX_s100
                                 + 0.684966 w_k1_IBM_s100
                                 + 0.718288 w_k1_INTC_s100
                                 + 0.780688 w_k1_KO_s100
                                 + 0.783191 w_k1_MAT_s100
                                 + 0.858887 w_k1_MSFT_s100
                                 + 0.930722 w_k1_PG_s100 >= 1
 z_k1_s1001_constraint:          var_1 + z_k1_s1001 + 1.14616 w_k1_ADM_s100
                                 + 1.14892 w_k1_AXP_s100
                                 + 1.11013 w_k1_DIS_s100
                                 + 1.14906 w_k1_FDX_s100
                                 + 0.912668 w_k1_IBM_s100
                                 + 1.08581 w_k1_INTC_s100
                                 + 0.924113 w_k1_KO_s100
                                 + 0.758736 w_k1_MAT_s100
                                 + 1.1439 w_k1_MSFT_s100 + 1.03937 w_k1_PG_s100
                                 >= 1
 z_k1_s1002_constraint:          var_1 + z_k1_s1002 + 0.833399 w_k1_ADM_s100
                                 + 0.661357 w_k1_AXP_s100
                                 + 0.86013 w_k1_DIS_s100
                                 + 0.741155 w_k1_FDX_s100
                                 + 1.27804 w_k1_IBM_s100
                                 + 0.816979 w_k1_INTC_s100
                                 + 1.11038 w_k1_KO_s100 + 1.01044 w_k1_MAT_s100
                                 + 1.04 w_k1_MSFT_s100 + 1.10197 w_k1_PG_s100
                                 >= 1
 z_k1_s1003_constraint:          var_1 + z_k1_s1003 + 1.38675 w_k1_ADM_s100
                                 + 1.27767 w_k1_AXP_s100
                                 + 1.08332 w_k1_DIS_s100
                                 + 1.11317 w_k1_FDX_s100
                                 + 1.12129 w_k1_IBM_s100
                                 + 1.07099 w_k1_INTC_s100
                                 + 1.15421 w_k1_KO_s100 + 1.14931 w_k1_MAT_s100
                                 + 1.19275 w_k1_MSFT_s100
                                 + 1.12439 w_k1_PG_s100 >= 1
 z_k1_s1004_constraint:          var_1 + z_k1_s1004 + 0.875101 w_k1_ADM_s100
                                 + 1.1249 w_k1_AXP_s100
                                 + 0.892641 w_k1_DIS_s100
                                 + 0.871399 w_k1_FDX_s100
                                 + 1.15511 w_k1_IBM_s100
                                 + 0.890099 w_k1_INTC_s100
                                 + 0.989481 w_k1_KO_s100
                                 + 1.22026 w_k1_MAT_s100
                                 + 0.965493 w_k1_MSFT_s100
                                 + 1.19357 w_k1_PG_s100 >= 1
 z_k1_s1005_constraint:          var_1 + z_k1_s1005 + 1.15756 w_k1_ADM_s100
                                 + 1.28991 w_k1_AXP_s100
                                 + 1.53284 w_k1_DIS_s100
                                 + 1.09007 w_k1_FDX_s100
                                 + 1.43273 w_k1_IBM_s100
                                 + 1.19107 w_k1_INTC_s100
                                 + 1.50227 w_k1_KO_s100 + 1.33937 w_k1_MAT_s100
                                 + 1.2996 w_k1_MSFT_s100 + 1.33928 w_k1_PG_s100
                                 >= 1
 z_k1_s1006_constraint:          var_1 + z_k1_s1006 + 1.16742 w_k1_ADM_s100
                                 + 1.57235 w_k1_AXP_s100
                                 + 1.58159 w_k1_DIS_s100
                                 + 1.45481 w_k1_FDX_s100
                                 + 0.971422 w_k1_IBM_s100
                                 + 1.22388 w_k1_INTC_s100
                                 + 1.24787 w_k1_KO_s100 + 1.17015 w_k1_MAT_s100
                                 + 1.4068 w_k1_MSFT_s100 + 1.2202 w_k1_PG_s100
                                 >= 1
 z_k1_s1007_constraint:          var_1 + z_k1_s1007 + 0.646185 w_k1_ADM_s100
                                 + 0.653023 w_k1_AXP_s100
                                 + 0.691192 w_k1_DIS_s100
                                 + 0.708704 w_k1_FDX_s100
                                 + 0.91708 w_k1_IBM_s100
                                 + 0.917772 w_k1_INTC_s100
                                 + 0.876669 w_k1_KO_s100
                                 + 0.788558 w_k1_MAT_s100
                                 + 0.914363 w_k1_MSFT_s100
                                 + 0.896224 w_k1_PG_s100 >= 1
 z_k1_s1010_constraint:          var_1 + z_k1_s1010 + 0.854979 w_k1_ADM_s101
                                 + 0.688709 w_k1_AXP_s101
                                 + 0.687645 w_k1_DIS_s101
                                 + 0.946599 w_k1_FDX_s101
                                 + 0.616242 w_k1_IBM_s101
                                 + 0.99133 w_k1_INTC_s101
                                 + 0.625639 w_k1_KO_s101
                                 + 0.846499 w_k1_MAT_s101
                                 + 0.938008 w_k1_MSFT_s101
                                 + 0.854528 w_k1_PG_s101 >= 1
 z_k1_s1011_constraint:          var_1 + z_k1_s1011 + 1.42973 w_k1_ADM_s101
                                 + 1.3803 w_k1_AXP_s101 + 1.25606 w_k1_DIS_s101
                                 + 1.37529 w_k1_FDX_s101
                                 + 0.873892 w_k1_IBM_s101
                                 + 1.27572 w_k1_INTC_s101
                                 + 0.989062 w_k1_KO_s101
                                 + 0.79223 w_k1_MAT_s101
                                 + 1.17611 w_k1_MSFT_s101
                                 + 1.21335 w_k1_PG_s101 >= 1
 z_k1_s1012_constraint:          var_1 + z_k1_s1012 + 1.29126 w_k1_ADM_s101
                                 + 1.77641 w_k1_AXP_s101
                                 + 2.14888 w_k1_DIS_s101
                                 + 1.77707 w_k1_FDX_s101
                                 + 1.25869 w_k1_IBM_s101
                                 + 1.66091 w_k1_INTC_s101
                                 + 1.55479 w_k1_KO_s101 + 1.57146 w_k1_MAT_s101
                                 + 1.74225 w_k1_MSFT_s101
                                 + 1.55005 w_k1_PG_s101 >= 1
 z_k1_s1013_constraint:          var_1 + z_k1_s1013 + 1.04193 w_k1_ADM_s101
                                 + 0.778646 w_k1_AXP_s101
                                 + 1.09504 w_k1_DIS_s101
                                 + 1.16033 w_k1_FDX_s101
                                 + 0.829612 w_k1_IBM_s101
                                 + 1.10506 w_k1_INTC_s101
                                 + 0.964483 w_k1_KO_s101
                                 + 0.858512 w_k1_MAT_s101
                                 + 1.24784 w_k1_MSFT_s101
                                 + 1.00393 w_k1_PG_s101 >= 1
 z_k1_s1014_constraint:          var_1 + z_k1_s1014 + 1.1513 w_k1_ADM_s101
                                 + 0.872406 w_k1_AXP_s101
                                 + 1.48629 w_k1_DIS_s101
                                 + 1.37182 w_k1_FDX_s101
                                 + 1.02631 w_k1_IBM_s101
                                 + 1.34638 w_k1_INTC_s101
                                 + 1.19433 w_k1_KO_s101
                                 + 0.914296 w_k1_MAT_s101
                                 + 1.34223 w_k1_MSFT_s101
                                 + 1.21134 w_k1_PG_s101 >= 1
 z_k1_s1015_constraint:          var_1 + z_k1_s1015 + 0.678142 w_k1_ADM_s101
                                 + 0.579887 w_k1_AXP_s101
                                 + 0.643202 w_k1_DIS_s101
                                 + 0.771575 w_k1_FDX_s101
                                 + 0.912967 w_k1_IBM_s101
                                 + 0.840638 w_k1_INTC_s101
                                 + 0.809345 w_k1_KO_s101
                                 + 0.852165 w_k1_MAT_s101
                                 + 0.75391 w_k1_MSFT_s101
                                 + 0.834337 w_k1_PG_s101 >= 1
 z_k1_s1016_constraint:          var_1 + z_k1_s1016 + 0.984194 w_k1_ADM_s101
                                 + 1.10072 w_k1_AXP_s101
                                 + 0.930127 w_k1_DIS_s101
                                 + 1.08504 w_k1_FDX_s101
                                 + 0.973401 w_k1_IBM_s101
                                 + 0.860982 w_k1_INTC_s101
                                 + 0.883932 w_k1_KO_s101
                                 + 1.08331 w_k1_MAT_s101
                                 + 0.884187 w_k1_MSFT_s101
                                 + 1.05586 w_k1_PG_s101 >= 1
 z_k1_s1017_constraint:          var_1 + z_k1_s1017 + 1.13448 w_k1_ADM_s101
                                 + 1.10277 w_k1_AXP_s101
                                 + 1.06946 w_k1_DIS_s101
                                 + 1.04125 w_k1_FDX_s101
                                 + 1.63263 w_k1_IBM_s101
                                 + 0.949128 w_k1_INTC_s101
                                 + 1.2024 w_k1_KO_s101 + 1.72371 w_k1_MAT_s101
                                 + 0.927362 w_k1_MSFT_s101
                                 + 1.09385 w_k1_PG_s101 >= 1
 z_k1_s1020_constraint:          var_1 + z_k1_s1020 + 1.17934 w_k1_ADM_s102
                                 + 1.25934 w_k1_AXP_s102
                                 + 1.15005 w_k1_DIS_s102
                                 + 1.11349 w_k1_FDX_s102
                                 + 1.43331 w_k1_IBM_s102
                                 + 1.10652 w_k1_INTC_s102
                                 + 1.19112 w_k1_KO_s102 + 1.04058 w_k1_MAT_s102
                                 + 1.12794 w_k1_MSFT_s102
                                 + 1.07439 w_k1_PG_s102 >= 1
 z_k1_s1021_constraint:          var_1 + z_k1_s1021 + 1.04803 w_k1_ADM_s102
                                 + 1.28844 w_k1_AXP_s102
                                 + 1.14003 w_k1_DIS_s102
                                 + 1.28534 w_k1_FDX_s102
                                 + 1.03611 w_k1_IBM_s102
                                 + 1.20784 w_k1_INTC_s102
                                 + 1.10259 w_k1_KO_s102 + 1.12872 w_k1_MAT_s102
                                 + 1.20036 w_k1_MSFT_s102
                                 + 1.04358 w_k1_PG_s102 >= 1
 z_k1_s1022_constraint:          var_1 + z_k1_s1022 + 0.981769 w_k1_ADM_s102
                                 + 0.832868 w_k1_AXP_s102
                                 + 1.05228 w_k1_DIS_s102
                                 + 1.04396 w_k1_FDX_s102
                                 + 0.840773 w_k1_IBM_s102
                                 + 1.1143 w_k1_INTC_s102
                                 + 0.917365 w_k1_KO_s102
                                 + 0.634034 w_k1_MAT_s102
                                 + 1.03489 w_k1_MSFT_s102
                                 + 0.927563 w_k1_PG_s102 >= 1
 z_k1_s1023_constraint:          var_1 + z_k1_s1023 + 0.845454 w_k1_ADM_s102
                                 + 1.0498 w_k1_AXP_s102
                                 + 0.917555 w_k1_DIS_s102
                                 + 0.931757 w_k1_FDX_s102
                                 + 1.15286 w_k1_IBM_s102
                                 + 0.8585 w_k1_INTC_s102 + 1.03164 w_k1_KO_s102
                                 + 1.13203 w_k1_MAT_s102
                                 + 0.944995 w_k1_MSFT_s102
                                 + 0.89296 w_k1_PG_s102 >= 1
 z_k1_s1024_constraint:          var_1 + z_k1_s1024 + 0.960204 w_k1_ADM_s102
                                 + 0.972059 w_k1_AXP_s102
                                 + 0.729904 w_k1_DIS_s102
                                 + 0.992238 w_k1_FDX_s102
                                 + 0.735622 w_k1_IBM_s102
                                 + 0.886032 w_k1_INTC_s102
                                 + 0.77831 w_k1_KO_s102
                                 + 0.659259 w_k1_MAT_s102
                                 + 0.813995 w_k1_MSFT_s102
                                 + 0.901829 w_k1_PG_s102 >= 1
 z_k1_s1025_constraint:          var_1 + z_k1_s1025 + 1.05378 w_k1_ADM_s102
                                 + 1.16287 w_k1_AXP_s102
                                 + 1.65522 w_k1_DIS_s102
                                 + 1.29408 w_k1_FDX_s102 + 1.5482 w_k1_IBM_s102
                                 + 1.14212 w_k1_INTC_s102
                                 + 1.53186 w_k1_KO_s102 + 1.60125 w_k1_MAT_s102
                                 + 1.27876 w_k1_MSFT_s102
                                 + 1.18563 w_k1_PG_s102 >= 1
 z_k1_s1026_constraint:          var_1 + z_k1_s1026 + 0.788743 w_k1_ADM_s102
                                 + 0.71502 w_k1_AXP_s102
                                 + 0.61923 w_k1_DIS_s102
                                 + 0.795266 w_k1_FDX_s102
                                 + 0.852209 w_k1_IBM_s102
                                 + 0.826483 w_k1_INTC_s102
                                 + 0.814076 w_k1_KO_s102
                                 + 0.729672 w_k1_MAT_s102
                                 + 0.79443 w_k1_MSFT_s102
                                 + 0.853668 w_k1_PG_s102 >= 1
 z_k1_s1027_constraint:          var_1 + z_k1_s1027 + 1.20088 w_k1_ADM_s102
                                 + 1.31173 w_k1_AXP_s102
                                 + 1.53857 w_k1_DIS_s102
                                 + 1.39837 w_k1_FDX_s102
                                 + 0.973194 w_k1_IBM_s102
                                 + 1.21132 w_k1_INTC_s102
                                 + 1.15095 w_k1_KO_s102
                                 + 0.864903 w_k1_MAT_s102
                                 + 1.32905 w_k1_MSFT_s102
                                 + 0.986964 w_k1_PG_s102 >= 1
 z_k1_s1030_constraint:          var_1 + z_k1_s1030 + 0.889661 w_k1_ADM_s103
                                 + 1.0325 w_k1_AXP_s103
                                 + 0.778173 w_k1_DIS_s103
                                 + 0.896525 w_k1_FDX_s103
                                 + 0.908085 w_k1_IBM_s103
                                 + 0.844728 w_k1_INTC_s103
                                 + 0.853343 w_k1_KO_s103
                                 + 1.01367 w_k1_MAT_s103
                                 + 0.895487 w_k1_MSFT_s103
                                 + 0.855537 w_k1_PG_s103 >= 1
 z_k1_s1031_constraint:          var_1 + z_k1_s1031 + 1.38992 w_k1_ADM_s103
                                 + 1.55186 w_k1_AXP_s103
                                 + 1.68445 w_k1_DIS_s103
                                 + 1.30223 w_k1_FDX_s103
                                 + 1.51542 w_k1_IBM_s103
                                 + 1.46434 w_k1_INTC_s103
                                 + 1.52062 w_k1_KO_s103 + 1.83098 w_k1_MAT_s103
                                 + 1.39025 w_k1_MSFT_s103
                                 + 1.17308 w_k1_PG_s103 >= 1
 z_k1_s1032_constraint:          var_1 + z_k1_s1032 + 1.04173 w_k1_ADM_s103
                                 + 0.92894 w_k1_AXP_s103
                                 + 1.10603 w_k1_DIS_s103
                                 + 1.01971 w_k1_FDX_s103
                                 + 0.865239 w_k1_IBM_s103
                                 + 1.10738 w_k1_INTC_s103
                                 + 1.01043 w_k1_KO_s103
                                 + 0.677348 w_k1_MAT_s103
                                 + 1.06724 w_k1_MSFT_s103
                                 + 0.900038 w_k1_PG_s103 >= 1
 z_k1_s1033_constraint:          var_1 + z_k1_s1033 + 1.44728 w_k1_ADM_s103
                                 + 1.49101 w_k1_AXP_s103
                                 + 1.25491 w_k1_DIS_s103
                                 + 1.34495 w_k1_FDX_s103
                                 + 1.08543 w_k1_IBM_s103
                                 + 1.27568 w_k1_INTC_s103
                                 + 1.16168 w_k1_KO_s103 + 1.48648 w_k1_MAT_s103
                                 + 1.29411 w_k1_MSFT_s103
                                 + 1.05919 w_k1_PG_s103 >= 1
 z_k1_s1034_constraint:          var_1 + z_k1_s1034 + 0.956743 w_k1_ADM_s103
                                 + 0.99918 w_k1_AXP_s103
                                 + 1.09686 w_k1_DIS_s103
                                 + 0.949249 w_k1_FDX_s103
                                 + 1.22182 w_k1_IBM_s103
                                 + 1.06144 w_k1_INTC_s103 + 1.1114 w_k1_KO_s103
                                 + 1.37938 w_k1_MAT_s103
                                 + 1.06189 w_k1_MSFT_s103
                                 + 1.01505 w_k1_PG_s103 >= 1
 z_k1_s1035_constraint:          var_1 + z_k1_s1035 + 1.43636 w_k1_ADM_s103
                                 + 1.24515 w_k1_AXP_s103 + 1.2264 w_k1_DIS_s103
                                 + 1.33638 w_k1_FDX_s103
                                 + 0.739084 w_k1_IBM_s103
                                 + 1.14399 w_k1_INTC_s103
                                 + 0.972727 w_k1_KO_s103
                                 + 0.907733 w_k1_MAT_s103
                                 + 1.11822 w_k1_MSFT_s103
                                 + 1.00407 w_k1_PG_s103 >= 1
 z_k1_s1036_constraint:          var_1 + z_k1_s1036 + 0.833867 w_k1_ADM_s103
                                 + 0.613727 w_k1_AXP_s103
                                 + 0.631962 w_k1_DIS_s103
                                 + 0.76017 w_k1_FDX_s103
                                 + 0.777984 w_k1_IBM_s103
                                 + 0.883813 w_k1_INTC_s103
                                 + 0.873919 w_k1_KO_s103
                                 + 0.687087 w_k1_MAT_s103
                                 + 0.802325 w_k1_MSFT_s103
                                 + 0.738491 w_k1_PG_s103 >= 1
 z_k1_s1037_constraint:          var_1 + z_k1_s1037 + 0.915775 w_k1_ADM_s103
                                 + 0.956897 w_k1_AXP_s103
                                 + 0.797709 w_k1_DIS_s103
                                 + 0.942187 w_k1_FDX_s103
                                 + 0.652659 w_k1_IBM_s103
                                 + 1.23763 w_k1_INTC_s103
                                 + 0.796867 w_k1_KO_s103
                                 + 0.73828 w_k1_MAT_s103
                                 + 0.958679 w_k1_MSFT_s103
                                 + 0.833162 w_k1_PG_s103 >= 1
 z_k1_s1040_constraint:          var_1 + z_k1_s1040 + 1.28409 w_k1_ADM_s104
                                 + 1.1925 w_k1_AXP_s104 + 1.13632 w_k1_DIS_s104
                                 + 1.15032 w_k1_FDX_s104
                                 + 0.939743 w_k1_IBM_s104
                                 + 1.1721 w_k1_INTC_s104 + 1.0629 w_k1_KO_s104
                                 + 0.948948 w_k1_MAT_s104
                                 + 1.07879 w_k1_MSFT_s104
                                 + 1.05355 w_k1_PG_s104 >= 1
 z_k1_s1041_constraint:          var_1 + z_k1_s1041 + 0.835185 w_k1_ADM_s104
                                 + 0.963427 w_k1_AXP_s104
                                 + 0.846543 w_k1_DIS_s104
                                 + 0.8776 w_k1_FDX_s104 + 1.10695 w_k1_IBM_s104
                                 + 0.925192 w_k1_INTC_s104
                                 + 0.936745 w_k1_KO_s104
                                 + 1.10809 w_k1_MAT_s104
                                 + 0.717393 w_k1_MSFT_s104
                                 + 0.923459 w_k1_PG_s104 >= 1
 z_k1_s1042_constraint:          var_1 + z_k1_s1042 + 0.838355 w_k1_ADM_s104
                                 + 0.750738 w_k1_AXP_s104
                                 + 0.767578 w_k1_DIS_s104
                                 + 0.897566 w_k1_FDX_s104
                                 + 0.808203 w_k1_IBM_s104
                                 + 0.777786 w_k1_INTC_s104
                                 + 0.874471 w_k1_KO_s104
                                 + 0.77621 w_k1_MAT_s104
                                 + 0.757719 w_k1_MSFT_s104
                                 + 0.89302 w_k1_PG_s104 >= 1
 z_k1_s1043_constraint:          var_1 + z_k1_s1043 + 0.91608 w_k1_ADM_s104
                                 + 0.790111 w_k1_AXP_s104
                                 + 0.981593 w_k1_DIS_s104
                                 + 0.902339 w_k1_FDX_s104
                                 + 1.41849 w_k1_IBM_s104
                                 + 0.844008 w_k1_INTC_s104
                                 + 1.29843 w_k1_KO_s104 + 1.5295 w_k1_MAT_s104
                                 + 0.948713 w_k1_MSFT_s104
                                 + 0.992089 w_k1_PG_s104 >= 1
 z_k1_s1044_constraint:          var_1 + z_k1_s1044 + 1.44594 w_k1_ADM_s104
                                 + 1.35307 w_k1_AXP_s104 + 1.25 w_k1_DIS_s104
                                 + 1.33741 w_k1_FDX_s104
                                 + 1.29607 w_k1_IBM_s104
                                 + 1.07458 w_k1_INTC_s104 + 1.2203 w_k1_KO_s104
                                 + 1.53019 w_k1_MAT_s104
                                 + 1.01982 w_k1_MSFT_s104
                                 + 1.16102 w_k1_PG_s104 >= 1
 z_k1_s1045_constraint:          var_1 + z_k1_s1045 + 1.26918 w_k1_ADM_s104
                                 + 1.18211 w_k1_AXP_s104
                                 + 1.15616 w_k1_DIS_s104
                                 + 1.08526 w_k1_FDX_s104 + 1.2207 w_k1_IBM_s104
                                 + 0.982217 w_k1_INTC_s104
                                 + 1.1473 w_k1_KO_s104 + 1.16901 w_k1_MAT_s104
                                 + 1.01289 w_k1_MSFT_s104
                                 + 1.02791 w_k1_PG_s104 >= 1
 z_k1_s1046_constraint:          var_1 + z_k1_s1046 + 1.38743 w_k1_ADM_s104
                                 + 1.24271 w_k1_AXP_s104
                                 + 1.74756 w_k1_DIS_s104
                                 + 1.44535 w_k1_FDX_s104
                                 + 1.25629 w_k1_IBM_s104
                                 + 1.05474 w_k1_INTC_s104
                                 + 1.53234 w_k1_KO_s104 + 1.27974 w_k1_MAT_s104
                                 + 1.30946 w_k1_MSFT_s104
                                 + 1.20638 w_k1_PG_s104 >= 1
 z_k1_s1047_constraint:          var_1 + z_k1_s1047 + 0.897925 w_k1_ADM_s104
                                 + 0.801179 w_k1_AXP_s104
                                 + 1.05664 w_k1_DIS_s104 + 1.0531 w_k1_FDX_s104
                                 + 0.893262 w_k1_IBM_s104
                                 + 0.924418 w_k1_INTC_s104
                                 + 1.15494 w_k1_KO_s104
                                 + 0.807285 w_k1_MAT_s104
                                 + 1.05613 w_k1_MSFT_s104
                                 + 0.977283 w_k1_PG_s104 >= 1
 z_k1_s1050_constraint:          var_1 + z_k1_s1050 + 1.09994 w_k1_ADM_s105
                                 + 1.25609 w_k1_AXP_s105
                                 + 1.20979 w_k1_DIS_s105
                                 + 1.06169 w_k1_FDX_s105
                                 + 1.26468 w_k1_IBM_s105
                                 + 1.12124 w_k1_INTC_s105
                                 + 1.20815 w_k1_KO_s105 + 1.3999 w_k1_MAT_s105
                                 + 1.02454 w_k1_MSFT_s105
                                 + 1.14644 w_k1_PG_s105 >= 1
 z_k1_s1051_constraint:          var_1 + z_k1_s1051 + 1.24687 w_k1_ADM_s105
                                 + 1.0626 w_k1_AXP_s105 + 1.05605 w_k1_DIS_s105
                                 + 1.06317 w_k1_FDX_s105
                                 + 0.955851 w_k1_IBM_s105
                                 + 1.10537 w_k1_INTC_s105 + 1.1469 w_k1_KO_s105
                                 + 1.11904 w_k1_MAT_s105
                                 + 1.05892 w_k1_MSFT_s105
                                 + 1.04019 w_k1_PG_s105 >= 1
 z_k1_s1052_constraint:          var_1 + z_k1_s1052 + 0.949714 w_k1_ADM_s105
                                 + 0.805565 w_k1_AXP_s105
                                 + 0.878075 w_k1_DIS_s105
                                 + 0.802017 w_k1_FDX_s105
                                 + 1.28284 w_k1_IBM_s105
                                 + 0.928515 w_k1_INTC_s105
                                 + 1.05793 w_k1_KO_s105 + 1.28093 w_k1_MAT_s105
                                 + 0.934955 w_k1_MSFT_s105
                                 + 0.976414 w_k1_PG_s105 >= 1
 z_k1_s1053_constraint:          var_1 + z_k1_s1053 + 0.905436 w_k1_ADM_s105
                                 + 1.33049 w_k1_AXP_s105
                                 + 1.34919 w_k1_DIS_s105
                                 + 1.05605 w_k1_FDX_s105 + 1.6812 w_k1_IBM_s105
                                 + 1.07702 w_k1_INTC_s105
                                 + 1.47829 w_k1_KO_s105 + 1.93109 w_k1_MAT_s105
                                 + 1.05937 w_k1_MSFT_s105
                                 + 1.23055 w_k1_PG_s105 >= 1
 z_k1_s1054_constraint:          var_1 + z_k1_s1054 + 0.862582 w_k1_ADM_s105
                                 + 0.84296 w_k1_AXP_s105
                                 + 0.783963 w_k1_DIS_s105
                                 + 0.817792 w_k1_FDX_s105
                                 + 0.906311 w_k1_IBM_s105
                                 + 0.93921 w_k1_INTC_s105
                                 + 0.971888 w_k1_KO_s105
                                 + 1.04374 w_k1_MAT_s105
                                 + 0.8941 w_k1_MSFT_s105
                                 + 0.895708 w_k1_PG_s105 >= 1
 z_k1_s1055_constraint:          var_1 + z_k1_s1055 + 1.03985 w_k1_ADM_s105
                                 + 0.777884 w_k1_AXP_s105
                                 + 0.736008 w_k1_DIS_s105
                                 + 0.905395 w_k1_FDX_s105
                                 + 0.621392 w_k1_IBM_s105
                                 + 0.868982 w_k1_INTC_s105
                                 + 0.68805 w_k1_KO_s105
                                 + 0.587828 w_k1_MAT_s105
                                 + 0.866449 w_k1_MSFT_s105
                                 + 0.850816 w_k1_PG_s105 >= 1
 z_k1_s1056_constraint:          var_1 + z_k1_s1056 + 1.28729 w_k1_ADM_s105
                                 + 1.54285 w_k1_AXP_s105
                                 + 1.54741 w_k1_DIS_s105
                                 + 1.48249 w_k1_FDX_s105
                                 + 1.04354 w_k1_IBM_s105
                                 + 1.22039 w_k1_INTC_s105
                                 + 1.37736 w_k1_KO_s105 + 1.57633 w_k1_MAT_s105
                                 + 1.34107 w_k1_MSFT_s105
                                 + 1.36889 w_k1_PG_s105 >= 1
 z_k1_s1057_constraint:          var_1 + z_k1_s1057 + 1.52679 w_k1_ADM_s105
                                 + 1.5131 w_k1_AXP_s105 + 1.13236 w_k1_DIS_s105
                                 + 1.35756 w_k1_FDX_s105
                                 + 0.821972 w_k1_IBM_s105
                                 + 1.25434 w_k1_INTC_s105
                                 + 1.07846 w_k1_KO_s105
                                 + 0.854497 w_k1_MAT_s105
                                 + 1.08827 w_k1_MSFT_s105
                                 + 1.16627 w_k1_PG_s105 >= 1
 z_k1_s1060_constraint:          var_1 + z_k1_s1060 + 0.943513 w_k1_ADM_s106
                                 + 0.951119 w_k1_AXP_s106
                                 + 1.04549 w_k1_DIS_s106
                                 + 1.06704 w_k1_FDX_s106
                                 + 0.848453 w_k1_IBM_s106
                                 + 1.05968 w_k1_INTC_s106
                                 + 0.912308 w_k1_KO_s106
                                 + 0.847539 w_k1_MAT_s106
                                 + 1.09483 w_k1_MSFT_s106
                                 + 0.840166 w_k1_PG_s106 >= 1
 z_k1_s1061_constraint:          var_1 + z_k1_s1061 + 1.01532 w_k1_ADM_s106
                                 + 1.09223 w_k1_AXP_s106
                                 + 0.965071 w_k1_DIS_s106
                                 + 0.99145 w_k1_FDX_s106
                                 + 0.999992 w_k1_IBM_s106
                                 + 1.02903 w_k1_INTC_s106
                                 + 0.906538 w_k1_KO_s106
                                 + 1.17185 w_k1_MAT_s106
                                 + 0.970307 w_k1_MSFT_s106
                                 + 0.908218 w_k1_PG_s106 >= 1
 z_k1_s1062_constraint:          var_1 + z_k1_s1062 + 1.2003 w_k1_ADM_s106
                                 + 0.985372 w_k1_AXP_s106
                                 + 1.10968 w_k1_DIS_s106
                                 + 0.987661 w_k1_FDX_s106
                                 + 1.22977 w_k1_IBM_s106
                                 + 0.853452 w_k1_INTC_s106
                                 + 1.21849 w_k1_KO_s106 + 1.34369 w_k1_MAT_s106
                                 + 0.986245 w_k1_MSFT_s106
                                 + 0.930174 w_k1_PG_s106 >= 1
 z_k1_s1063_constraint:          var_1 + z_k1_s1063 + 1.04957 w_k1_ADM_s106
                                 + 0.831124 w_k1_AXP_s106
                                 + 0.906711 w_k1_DIS_s106
                                 + 0.980363 w_k1_FDX_s106
                                 + 0.684811 w_k1_IBM_s106
                                 + 0.791168 w_k1_INTC_s106
                                 + 0.798265 w_k1_KO_s106
                                 + 0.857842 w_k1_MAT_s106
                                 + 0.95701 w_k1_MSFT_s106
                                 + 0.831181 w_k1_PG_s106 >= 1
 z_k1_s1064_constraint:          var_1 + z_k1_s1064 + 1.3027 w_k1_ADM_s106
                                 + 1.33235 w_k1_AXP_s106
                                 + 1.58731 w_k1_DIS_s106
                                 + 1.42606 w_k1_FDX_s106
                                 + 1.03196 w_k1_IBM_s106
                                 + 1.27082 w_k1_INTC_s106
                                 + 1.31459 w_k1_KO_s106 + 1.31314 w_k1_MAT_s106
                                 + 1.34646 w_k1_MSFT_s106
                                 + 1.14435 w_k1_PG_s106 >= 1
 z_k1_s1065_constraint:          var_1 + z_k1_s1065 + 1.00969 w_k1_ADM_s106
                                 + 0.721003 w_k1_AXP_s106
                                 + 0.657687 w_k1_DIS_s106
                                 + 0.738316 w_k1_FDX_s106
                                 + 0.814539 w_k1_IBM_s106
                                 + 0.636955 w_k1_INTC_s106
                                 + 0.745131 w_k1_KO_s106
                                 + 0.809674 w_k1_MAT_s106
                                 + 0.718284 w_k1_MSFT_s106
                                 + 0.771056 w_k1_PG_s106 >= 1
 z_k1_s1066_constraint:          var_1 + z_k1_s1066 + 0.991609 w_k1_ADM_s106
                                 + 0.746301 w_k1_AXP_s106
                                 + 0.784106 w_k1_DIS_s106
                                 + 0.896554 w_k1_FDX_s106
                                 + 0.812209 w_k1_IBM_s106
                                 + 0.866719 w_k1_INTC_s106
                                 + 0.898639 w_k1_KO_s106
                                 + 0.941365 w_k1_MAT_s106
                                 + 1.03125 w_k1_MSFT_s106
                                 + 0.849461 w_k1_PG_s106 >= 1
 z_k1_s1067_constraint:          var_1 + z_k1_s1067 + 1.43218 w_k1_ADM_s106
                                 + 1.36436 w_k1_AXP_s106
                                 + 1.39508 w_k1_DIS_s106
                                 + 1.18603 w_k1_FDX_s106
                                 + 1.43264 w_k1_IBM_s106
                                 + 0.937955 w_k1_INTC_s106
                                 + 1.41308 w_k1_KO_s106 + 1.73845 w_k1_MAT_s106
                                 + 1.1487 w_k1_MSFT_s106 + 1.11972 w_k1_PG_s106
                                 >= 1
 z_k1_s1070_constraint:          var_1 + z_k1_s1070 + 1.43082 w_k1_ADM_s107
                                 + 1.62213 w_k1_AXP_s107 + 1.9458 w_k1_DIS_s107
                                 + 1.45268 w_k1_FDX_s107 + 1.3848 w_k1_IBM_s107
                                 + 1.44962 w_k1_INTC_s107
                                 + 1.47345 w_k1_KO_s107 + 1.91586 w_k1_MAT_s107
                                 + 1.5496 w_k1_MSFT_s107 + 1.36246 w_k1_PG_s107
                                 >= 1
 z_k1_s1071_constraint:          var_1 + z_k1_s1071 + 1.20745 w_k1_ADM_s107
                                 + 1.128 w_k1_AXP_s107 + 1.21377 w_k1_DIS_s107
                                 + 1.00217 w_k1_FDX_s107
                                 + 1.70247 w_k1_IBM_s107
                                 + 1.24248 w_k1_INTC_s107
                                 + 1.12209 w_k1_KO_s107 + 1.18189 w_k1_MAT_s107
                                 + 1.05399 w_k1_MSFT_s107
                                 + 1.07852 w_k1_PG_s107 >= 1
 z_k1_s1072_constraint:          var_1 + z_k1_s1072 + 0.980685 w_k1_ADM_s107
                                 + 0.938984 w_k1_AXP_s107
                                 + 0.998386 w_k1_DIS_s107
                                 + 0.877313 w_k1_FDX_s107
                                 + 1.29598 w_k1_IBM_s107
                                 + 0.828214 w_k1_INTC_s107
                                 + 0.981775 w_k1_KO_s107
                                 + 1.13841 w_k1_MAT_s107
                                 + 0.839882 w_k1_MSFT_s107
                                 + 0.928914 w_k1_PG_s107 >= 1
 z_k1_s1073_constraint:          var_1 + z_k1_s1073 + 0.907241 w_k1_ADM_s107
                                 + 0.834811 w_k1_AXP_s107
                                 + 0.644016 w_k1_DIS_s107
                                 + 0.764622 w_k1_FDX_s107
                                 + 0.781521 w_k1_IBM_s107
                                 + 1.01898 w_k1_INTC_s107
                                 + 0.673556 w_k1_KO_s107
                                 + 0.546271 w_k1_MAT_s107
                                 + 0.765723 w_k1_MSFT_s107
                                 + 0.71231 w_k1_PG_s107 >= 1
 z_k1_s1074_constraint:          var_1 + z_k1_s1074 + 1.19369 w_k1_ADM_s107
                                 + 0.98417 w_k1_AXP_s107
                                 + 1.24371 w_k1_DIS_s107
                                 + 1.08149 w_k1_FDX_s107
                                 + 0.994098 w_k1_IBM_s107
                                 + 1.08745 w_k1_INTC_s107
                                 + 0.94153 w_k1_KO_s107 + 0.70506 w_k1_MAT_s107
                                 + 1.09388 w_k1_MSFT_s107
                                 + 0.921155 w_k1_PG_s107 >= 1
 z_k1_s1075_constraint:          var_1 + z_k1_s1075 + 0.926402 w_k1_ADM_s107
                                 + 0.711913 w_k1_AXP_s107
                                 + 0.851553 w_k1_DIS_s107
                                 + 0.870015 w_k1_FDX_s107
                                 + 0.830363 w_k1_IBM_s107
                                 + 0.908345 w_k1_INTC_s107
                                 + 0.857801 w_k1_KO_s107
                                 + 0.711237 w_k1_MAT_s107
                                 + 0.962715 w_k1_MSFT_s107
                                 + 0.888822 w_k1_PG_s107 >= 1
 z_k1_s1076_constraint:          var_1 + z_k1_s1076 + 1.2616 w_k1_ADM_s107
                                 + 1.16422 w_k1_AXP_s107
                                 + 0.981478 w_k1_DIS_s107
                                 + 1.12151 w_k1_FDX_s107
                                 + 0.904085 w_k1_IBM_s107
                                 + 0.976746 w_k1_INTC_s107
                                 + 0.834688 w_k1_KO_s107
                                 + 0.94739 w_k1_MAT_s107
                                 + 0.977718 w_k1_MSFT_s107
                                 + 0.945859 w_k1_PG_s107 >= 1
 z_k1_s1077_constraint:          var_1 + z_k1_s1077 + 1.38056 w_k1_ADM_s107
                                 + 1.38578 w_k1_AXP_s107
                                 + 1.51738 w_k1_DIS_s107 + 1.2621 w_k1_FDX_s107
                                 + 1.20894 w_k1_IBM_s107
                                 + 1.26641 w_k1_INTC_s107
                                 + 1.22419 w_k1_KO_s107 + 1.31446 w_k1_MAT_s107
                                 + 1.23593 w_k1_MSFT_s107
                                 + 1.13101 w_k1_PG_s107 >= 1
 z_k1_s1100_constraint:          var_1 + z_k1_s1100 + 0.949465 w_k1_ADM_s110
                                 + 1.10894 w_k1_AXP_s110
                                 + 1.08297 w_k1_DIS_s110
                                 + 1.12425 w_k1_FDX_s110
                                 + 1.21144 w_k1_IBM_s110
                                 + 1.10036 w_k1_INTC_s110
                                 + 1.03475 w_k1_KO_s110 + 1.34862 w_k1_MAT_s110
                                 + 1.08624 w_k1_MSFT_s110
                                 + 1.12375 w_k1_PG_s110 >= 1
 z_k1_s1101_constraint:          var_1 + z_k1_s1101 + 1.06486 w_k1_ADM_s110
                                 + 1.37575 w_k1_AXP_s110
                                 + 1.32843 w_k1_DIS_s110
                                 + 1.32244 w_k1_FDX_s110
                                 + 1.27416 w_k1_IBM_s110
                                 + 0.992798 w_k1_INTC_s110
                                 + 1.26282 w_k1_KO_s110 + 1.39819 w_k1_MAT_s110
                                 + 1.24721 w_k1_MSFT_s110 + 1.3009 w_k1_PG_s110
                                 >= 1
 z_k1_s1102_constraint:          var_1 + z_k1_s1102 + 0.747852 w_k1_ADM_s110
                                 + 0.93609 w_k1_AXP_s110
                                 + 0.854675 w_k1_DIS_s110
                                 + 0.831206 w_k1_FDX_s110
                                 + 1.56608 w_k1_IBM_s110
                                 + 0.847411 w_k1_INTC_s110
                                 + 1.10716 w_k1_KO_s110 + 1.18848 w_k1_MAT_s110
                                 + 0.841279 w_k1_MSFT_s110
                                 + 0.921465 w_k1_PG_s110 >= 1
 z_k1_s1103_constraint:          var_1 + z_k1_s1103 + 0.918906 w_k1_ADM_s110
                                 + 0.738391 w_k1_AXP_s110
                                 + 0.928766 w_k1_DIS_s110
                                 + 0.946446 w_k1_FDX_s110
                                 + 0.798662 w_k1_IBM_s110
                                 + 0.927291 w_k1_INTC_s110
                                 + 0.978521 w_k1_KO_s110
                                 + 0.558576 w_k1_MAT_s110
                                 + 0.937902 w_k1_MSFT_s110
                                 + 0.915109 w_k1_PG_s110 >= 1
 z_k1_s1104_constraint:          var_1 + z_k1_s1104 + 0.688343 w_k1_ADM_s110
                                 + 0.608737 w_k1_AXP_s110
                                 + 0.532219 w_k1_DIS_s110
                                 + 0.690151 w_k1_FDX_s110
                                 + 0.674728 w_k1_IBM_s110
                                 + 0.624035 w_k1_INTC_s110
                                 + 0.649926 w_k1_KO_s110
                                 + 0.509736 w_k1_MAT_s110
                                 + 0.603452 w_k1_MSFT_s110
                                 + 0.78142 w_k1_PG_s110 >= 1
 z_k1_s1105_constraint:          var_1 + z_k1_s1105 + 0.499943 w_k1_ADM_s110
                                 + 0.709672 w_k1_AXP_s110
                                 + 0.767715 w_k1_DIS_s110
                                 + 0.761713 w_k1_FDX_s110
                                 + 1.04952 w_k1_IBM_s110
                                 + 0.842378 w_k1_INTC_s110
                                 + 0.82562 w_k1_KO_s110
                                 + 0.899876 w_k1_MAT_s110
                                 + 0.733081 w_k1_MSFT_s110
                                 + 0.80807 w_k1_PG_s110 >= 1
 z_k1_s1106_constraint:          var_1 + z_k1_s1106 + 1.09816 w_k1_ADM_s110
                                 + 1.00193 w_k1_AXP_s110
                                 + 0.98751 w_k1_DIS_s110
                                 + 1.13609 w_k1_FDX_s110
                                 + 0.800175 w_k1_IBM_s110
                                 + 1.05532 w_k1_INTC_s110
                                 + 0.909985 w_k1_KO_s110
                                 + 0.878812 w_k1_MAT_s110
                                 + 0.968215 w_k1_MSFT_s110
                                 + 0.936988 w_k1_PG_s110 >= 1
 z_k1_s1107_constraint:          var_1 + z_k1_s1107 + 0.984667 w_k1_ADM_s110
                                 + 1.44205 w_k1_AXP_s110 + 1.6584 w_k1_DIS_s110
                                 + 1.25697 w_k1_FDX_s110
                                 + 1.77383 w_k1_IBM_s110
                                 + 1.15524 w_k1_INTC_s110
                                 + 1.38257 w_k1_KO_s110 + 1.77165 w_k1_MAT_s110
                                 + 1.15103 w_k1_MSFT_s110
                                 + 1.22171 w_k1_PG_s110 >= 1
 z_k1_s1110_constraint:          var_1 + z_k1_s1110 + 0.74295 w_k1_ADM_s111
                                 + 0.95945 w_k1_AXP_s111
                                 + 0.950861 w_k1_DIS_s111
                                 + 0.916961 w_k1_FDX_s111
                                 + 0.870589 w_k1_IBM_s111
                                 + 0.997894 w_k1_INTC_s111
                                 + 0.920346 w_k1_KO_s111
                                 + 0.73053 w_k1_MAT_s111
                                 + 0.931272 w_k1_MSFT_s111
                                 + 0.939063 w_k1_PG_s111 >= 1
 z_k1_s1111_constraint:          var_1 + z_k1_s1111 + 1.13361 w_k1_ADM_s111
                                 + 1.10415 w_k1_AXP_s111
                                 + 0.943702 w_k1_DIS_s111
                                 + 1.11461 w_k1_FDX_s111 + 0.8547 w_k1_IBM_s111
                                 + 1.09788 w_k1_INTC_s111
                                 + 0.976751 w_k1_KO_s111
                                 + 0.53421 w_k1_MAT_s111
                                 + 1.03715 w_k1_MSFT_s111
                                 + 0.902317 w_k1_PG_s111 >= 1
 z_k1_s1112_constraint:          var_1 + z_k1_s1112 + 0.909525 w_k1_ADM_s111
                                 + 1.04261 w_k1_AXP_s111
                                 + 0.943108 w_k1_DIS_s111
                                 + 0.854061 w_k1_FDX_s111
                                 + 1.27014 w_k1_IBM_s111
                                 + 0.941411 w_k1_INTC_s111
                                 + 1.09636 w_k1_KO_s111 + 1.08011 w_k1_MAT_s111
                                 + 0.93517 w_k1_MSFT_s111
                                 + 0.943181 w_k1_PG_s111 >= 1
 z_k1_s1113_constraint:          var_1 + z_k1_s1113 + 0.835077 w_k1_ADM_s111
                                 + 0.747569 w_k1_AXP_s111
                                 + 0.724772 w_k1_DIS_s111
                                 + 0.730529 w_k1_FDX_s111
                                 + 0.775064 w_k1_IBM_s111
                                 + 0.689709 w_k1_INTC_s111
                                 + 0.808213 w_k1_KO_s111
                                 + 0.71707 w_k1_MAT_s111
                                 + 0.753843 w_k1_MSFT_s111
                                 + 0.854462 w_k1_PG_s111 >= 1
 z_k1_s1114_constraint:          var_1 + z_k1_s1114 + 1.40094 w_k1_ADM_s111
                                 + 1.30706 w_k1_AXP_s111
                                 + 1.72511 w_k1_DIS_s111 + 1.2402 w_k1_FDX_s111
                                 + 1.61079 w_k1_IBM_s111
                                 + 0.93071 w_k1_INTC_s111 + 1.5729 w_k1_KO_s111
                                 + 1.10224 w_k1_MAT_s111
                                 + 1.12969 w_k1_MSFT_s111
                                 + 1.18053 w_k1_PG_s111 >= 1
 z_k1_s1115_constraint:          var_1 + z_k1_s1115 + 1.69535 w_k1_ADM_s111
                                 + 1.46543 w_k1_AXP_s111
                                 + 1.74619 w_k1_DIS_s111
                                 + 1.41686 w_k1_FDX_s111
                                 + 1.06611 w_k1_IBM_s111
                                 + 1.02708 w_k1_INTC_s111
                                 + 1.25981 w_k1_KO_s111 + 1.20829 w_k1_MAT_s111
                                 + 1.24898 w_k1_MSFT_s111
                                 + 1.22325 w_k1_PG_s111 >= 1
 z_k1_s1116_constraint:          var_1 + z_k1_s1116 + 1.04989 w_k1_ADM_s111
                                 + 1.00545 w_k1_AXP_s111
                                 + 1.22448 w_k1_DIS_s111
                                 + 1.02869 w_k1_FDX_s111
                                 + 1.12733 w_k1_IBM_s111
                                 + 0.97135 w_k1_INTC_s111
                                 + 1.10912 w_k1_KO_s111
                                 + 0.786259 w_k1_MAT_s111
                                 + 1.09214 w_k1_MSFT_s111
                                 + 1.08432 w_k1_PG_s111 >= 1
 z_k1_s1117_constraint:          var_1 + z_k1_s1117 + 1.16101 w_k1_ADM_s111
                                 + 1.15475 w_k1_AXP_s111 + 1.46 w_k1_DIS_s111
                                 + 1.21638 w_k1_FDX_s111
                                 + 1.05615 w_k1_IBM_s111
                                 + 0.986544 w_k1_INTC_s111
                                 + 1.31147 w_k1_KO_s111 + 1.04714 w_k1_MAT_s111
                                 + 1.21139 w_k1_MSFT_s111
                                 + 0.978974 w_k1_PG_s111 >= 1
 z_k1_s1120_constraint:          var_1 + z_k1_s1120 + 1.02354 w_k1_ADM_s112
                                 + 0.946169 w_k1_AXP_s112
                                 + 1.18986 w_k1_DIS_s112
                                 + 1.09649 w_k1_FDX_s112
                                 + 1.14853 w_k1_IBM_s112
                                 + 0.95857 w_k1_INTC_s112
                                 + 1.14006 w_k1_KO_s112 + 1.46786 w_k1_MAT_s112
                                 + 0.99367 w_k1_MSFT_s112
                                 + 1.12067 w_k1_PG_s112 >= 1
 z_k1_s1121_constraint:          var_1 + z_k1_s1121 + 0.870049 w_k1_ADM_s112
                                 + 0.994244 w_k1_AXP_s112
                                 + 1.26966 w_k1_DIS_s112
                                 + 1.06988 w_k1_FDX_s112
                                 + 1.65876 w_k1_IBM_s112
                                 + 1.13765 w_k1_INTC_s112
                                 + 1.39916 w_k1_KO_s112 + 1.62912 w_k1_MAT_s112
                                 + 1.02323 w_k1_MSFT_s112
                                 + 1.09275 w_k1_PG_s112 >= 1
 z_k1_s1122_constraint:          var_1 + z_k1_s1122 + 1.38812 w_k1_ADM_s112
                                 + 1.23484 w_k1_AXP_s112
                                 + 1.43339 w_k1_DIS_s112
                                 + 1.33596 w_k1_FDX_s112
                                 + 0.997225 w_k1_IBM_s112
                                 + 1.4065 w_k1_INTC_s112 + 1.34663 w_k1_KO_s112
                                 + 1.05114 w_k1_MAT_s112
                                 + 1.29288 w_k1_MSFT_s112
                                 + 1.16578 w_k1_PG_s112 >= 1
 z_k1_s1123_constraint:          var_1 + z_k1_s1123 + 0.993514 w_k1_ADM_s112
                                 + 0.628099 w_k1_AXP_s112
                                 + 0.788338 w_k1_DIS_s112
                                 + 0.918659 w_k1_FDX_s112
                                 + 0.664674 w_k1_IBM_s112
                                 + 0.941644 w_k1_INTC_s112
                                 + 0.789293 w_k1_KO_s112
                                 + 0.43934 w_k1_MAT_s112
                                 + 0.979941 w_k1_MSFT_s112
                                 + 0.89214 w_k1_PG_s112 >= 1
 z_k1_s1124_constraint:          var_1 + z_k1_s1124 + 0.851007 w_k1_ADM_s112
                                 + 1.03481 w_k1_AXP_s112
                                 + 0.882543 w_k1_DIS_s112
                                 + 0.87909 w_k1_FDX_s112
                                 + 1.30442 w_k1_IBM_s112
                                 + 1.03938 w_k1_INTC_s112
                                 + 1.10063 w_k1_KO_s112 + 1.47579 w_k1_MAT_s112
                                 + 0.906179 w_k1_MSFT_s112
                                 + 0.989671 w_k1_PG_s112 >= 1
 z_k1_s1125_constraint:          var_1 + z_k1_s1125 + 1.05692 w_k1_ADM_s112
                                 + 0.713444 w_k1_AXP_s112
                                 + 1.01822 w_k1_DIS_s112
                                 + 0.975477 w_k1_FDX_s112
                                 + 0.954878 w_k1_IBM_s112
                                 + 0.906721 w_k1_INTC_s112
                                 + 1.10347 w_k1_KO_s112
                                 + 0.746469 w_k1_MAT_s112
                                 + 1.07605 w_k1_MSFT_s112 + 1.0042 w_k1_PG_s112
                                 >= 1
 z_k1_s1126_constraint:          var_1 + z_k1_s1126 + 0.766254 w_k1_ADM_s112
                                 + 0.75472 w_k1_AXP_s112
                                 + 0.630851 w_k1_DIS_s112
                                 + 0.712794 w_k1_FDX_s112
                                 + 0.926976 w_k1_IBM_s112
                                 + 0.65643 w_k1_INTC_s112
                                 + 0.795718 w_k1_KO_s112
                                 + 1.48756 w_k1_MAT_s112
                                 + 0.718173 w_k1_MSFT_s112
                                 + 0.824563 w_k1_PG_s112 >= 1
 z_k1_s1127_constraint:          var_1 + z_k1_s1127 + 1.45217 w_k1_ADM_s112
                                 + 1.37259 w_k1_AXP_s112
                                 + 1.96417 w_k1_DIS_s112
                                 + 1.35305 w_k1_FDX_s112
                                 + 1.65246 w_k1_IBM_s112
                                 + 1.45607 w_k1_INTC_s112 + 1.6829 w_k1_KO_s112
                                 + 1.42283 w_k1_MAT_s112
                                 + 1.41381 w_k1_MSFT_s112
                                 + 1.21638 w_k1_PG_s112 >= 1
 z_k1_s1130_constraint:          var_1 + z_k1_s1130 + 0.882328 w_k1_ADM_s113
                                 + 1.19802 w_k1_AXP_s113
                                 + 0.832803 w_k1_DIS_s113
                                 + 1.09061 w_k1_FDX_s113
                                 + 0.802462 w_k1_IBM_s113
                                 + 1.04472 w_k1_INTC_s113
                                 + 0.833308 w_k1_KO_s113
                                 + 1.09423 w_k1_MAT_s113
                                 + 0.900678 w_k1_MSFT_s113
                                 + 1.0723 w_k1_PG_s113 >= 1
 z_k1_s1131_constraint:          var_1 + z_k1_s1131 + 0.740189 w_k1_ADM_s113
                                 + 0.903377 w_k1_AXP_s113
                                 + 0.827515 w_k1_DIS_s113
                                 + 0.918933 w_k1_FDX_s113
                                 + 1.13596 w_k1_IBM_s113
                                 + 0.955224 w_k1_INTC_s113
                                 + 1.00545 w_k1_KO_s113
                                 + 0.969882 w_k1_MAT_s113
                                 + 0.973071 w_k1_MSFT_s113
                                 + 0.956673 w_k1_PG_s113 >= 1
 z_k1_s1132_constraint:          var_1 + z_k1_s1132 + 1.04672 w_k1_ADM_s113
                                 + 1.25139 w_k1_AXP_s113
                                 + 1.27031 w_k1_DIS_s113
                                 + 1.31115 w_k1_FDX_s113
                                 + 0.958896 w_k1_IBM_s113
                                 + 1.20235 w_k1_INTC_s113
                                 + 1.00019 w_k1_KO_s113 + 1.21603 w_k1_MAT_s113
                                 + 1.18004 w_k1_MSFT_s113
                                 + 1.13063 w_k1_PG_s113 >= 1
 z_k1_s1133_constraint:          var_1 + z_k1_s1133 + 0.687775 w_k1_ADM_s113
                                 + 0.71936 w_k1_AXP_s113
                                 + 0.670024 w_k1_DIS_s113
                                 + 0.838407 w_k1_FDX_s113
                                 + 0.862303 w_k1_IBM_s113
                                 + 0.835139 w_k1_INTC_s113
                                 + 0.864477 w_k1_KO_s113
                                 + 0.890293 w_k1_MAT_s113
                                 + 0.792071 w_k1_MSFT_s113
                                 + 0.882402 w_k1_PG_s113 >= 1
 z_k1_s1134_constraint:          var_1 + z_k1_s1134 + 0.882937 w_k1_ADM_s113
                                 + 1.01818 w_k1_AXP_s113
                                 + 1.24663 w_k1_DIS_s113
                                 + 1.13561 w_k1_FDX_s113
                                 + 0.985702 w_k1_IBM_s113
                                 + 1.07862 w_k1_INTC_s113
                                 + 1.20724 w_k1_KO_s113 + 1.07377 w_k1_MAT_s113
                                 + 1.02168 w_k1_MSFT_s113 + 1.1066 w_k1_PG_s113
                                 >= 1
 z_k1_s1135_constraint:          var_1 + z_k1_s1135 + 0.748019 w_k1_ADM_s113
                                 + 0.761691 w_k1_AXP_s113
                                 + 0.794904 w_k1_DIS_s113
                                 + 0.910318 w_k1_FDX_s113
                                 + 0.740179 w_k1_IBM_s113
                                 + 1.04644 w_k1_INTC_s113
                                 + 0.80334 w_k1_KO_s113
                                 + 0.604988 w_k1_MAT_s113
                                 + 0.788253 w_k1_MSFT_s113
                                 + 0.895268 w_k1_PG_s113 >= 1
 z_k1_s1136_constraint:          var_1 + z_k1_s1136 + 0.820642 w_k1_ADM_s113
                                 + 1.08101 w_k1_AXP_s113
                                 + 1.12861 w_k1_DIS_s113
                                 + 0.967183 w_k1_FDX_s113
                                 + 1.44336 w_k1_IBM_s113
                                 + 0.898913 w_k1_INTC_s113
                                 + 1.26295 w_k1_KO_s113 + 1.55561 w_k1_MAT_s113
                                 + 0.956495 w_k1_MSFT_s113
                                 + 1.16013 w_k1_PG_s113 >= 1
 z_k1_s1137_constraint:          var_1 + z_k1_s1137 + 1.11462 w_k1_ADM_s113
                                 + 1.41666 w_k1_AXP_s113
                                 + 1.69823 w_k1_DIS_s113
                                 + 1.28983 w_k1_FDX_s113
                                 + 1.63688 w_k1_IBM_s113
                                 + 1.09246 w_k1_INTC_s113
                                 + 1.56697 w_k1_KO_s113 + 1.8685 w_k1_MAT_s113
                                 + 1.24786 w_k1_MSFT_s113 + 1.2179 w_k1_PG_s113
                                 >= 1
 z_k1_s1140_constraint:          var_1 + z_k1_s1140 + 1.07667 w_k1_ADM_s114
                                 + 1.02716 w_k1_AXP_s114
                                 + 0.816992 w_k1_DIS_s114
                                 + 1.09611 w_k1_FDX_s114
                                 + 0.676717 w_k1_IBM_s114
                                 + 1.07799 w_k1_INTC_s114
                                 + 0.942814 w_k1_KO_s114
                                 + 0.481113 w_k1_MAT_s114
                                 + 1.01809 w_k1_MSFT_s114
                                 + 0.957634 w_k1_PG_s114 >= 1
 z_k1_s1141_constraint:          var_1 + z_k1_s1141 + 1.46158 w_k1_ADM_s114
                                 + 1.4725 w_k1_AXP_s114 + 1.75225 w_k1_DIS_s114
                                 + 1.42179 w_k1_FDX_s114
                                 + 1.08845 w_k1_IBM_s114
                                 + 1.39766 w_k1_INTC_s114
                                 + 1.43105 w_k1_KO_s114 + 0.67117 w_k1_MAT_s114
                                 + 1.54946 w_k1_MSFT_s114
                                 + 1.15653 w_k1_PG_s114 >= 1
 z_k1_s1142_constraint:          var_1 + z_k1_s1142 + 1.15973 w_k1_ADM_s114
                                 + 1.39373 w_k1_AXP_s114
                                 + 1.10389 w_k1_DIS_s114
                                 + 1.10568 w_k1_FDX_s114
                                 + 1.36486 w_k1_IBM_s114
                                 + 1.24932 w_k1_INTC_s114
                                 + 1.25387 w_k1_KO_s114 + 1.18459 w_k1_MAT_s114
                                 + 1.03663 w_k1_MSFT_s114
                                 + 1.05569 w_k1_PG_s114 >= 1
 z_k1_s1143_constraint:          var_1 + z_k1_s1143 + 1.2291 w_k1_ADM_s114
                                 + 1.3007 w_k1_AXP_s114 + 1.30588 w_k1_DIS_s114
                                 + 1.13999 w_k1_FDX_s114 + 1.0769 w_k1_IBM_s114
                                 + 1.22539 w_k1_INTC_s114
                                 + 1.27026 w_k1_KO_s114 + 1.31703 w_k1_MAT_s114
                                 + 1.19593 w_k1_MSFT_s114
                                 + 1.17978 w_k1_PG_s114 >= 1
 z_k1_s1144_constraint:          var_1 + z_k1_s1144 + 0.893825 w_k1_ADM_s114
                                 + 0.934665 w_k1_AXP_s114
                                 + 1.03974 w_k1_DIS_s114
                                 + 0.957914 w_k1_FDX_s114
                                 + 1.18464 w_k1_IBM_s114
                                 + 0.894832 w_k1_INTC_s114
                                 + 1.09621 w_k1_KO_s114
                                 + 0.896067 w_k1_MAT_s114
                                 + 1.10715 w_k1_MSFT_s114
                                 + 1.10513 w_k1_PG_s114 >= 1
 z_k1_s1145_constraint:          var_1 + z_k1_s1145 + 0.726422 w_k1_ADM_s114
                                 + 0.862642 w_k1_AXP_s114
                                 + 0.720236 w_k1_DIS_s114
                                 + 0.769406 w_k1_FDX_s114
                                 + 0.90988 w_k1_IBM_s114
                                 + 0.963383 w_k1_INTC_s114
                                 + 0.856973 w_k1_KO_s114
                                 + 0.816987 w_k1_MAT_s114
                                 + 0.889326 w_k1_MSFT_s114
                                 + 0.818552 w_k1_PG_s114 >= 1
 z_k1_s1146_constraint:          var_1 + z_k1_s1146 + 0.822442 w_k1_ADM_s114
                                 + 1.01521 w_k1_AXP_s114
                                 + 0.82183 w_k1_DIS_s114
                                 + 0.92526 w_k1_FDX_s114
                                 + 0.796465 w_k1_IBM_s114
                                 + 0.959008 w_k1_INTC_s114
                                 + 0.891124 w_k1_KO_s114
                                 + 0.856257 w_k1_MAT_s114
                                 + 1.07716 w_k1_MSFT_s114
                                 + 0.968739 w_k1_PG_s114 >= 1
 z_k1_s1147_constraint:          var_1 + z_k1_s1147 + 1.21018 w_k1_ADM_s114
                                 + 1.16542 w_k1_AXP_s114
                                 + 1.28292 w_k1_DIS_s114
                                 + 1.25687 w_k1_FDX_s114
                                 + 0.715069 w_k1_IBM_s114
                                 + 1.19053 w_k1_INTC_s114
                                 + 1.03037 w_k1_KO_s114
                                 + 0.566622 w_k1_MAT_s114
                                 + 1.3525 w_k1_MSFT_s114 + 1.05353 w_k1_PG_s114
                                 >= 1
 z_k1_s1150_constraint:          var_1 + z_k1_s1150 + 1.0329 w_k1_ADM_s115
                                 + 0.834554 w_k1_AXP_s115
                                 + 1.02994 w_k1_DIS_s115
                                 + 1.07217 w_k1_FDX_s115
                                 + 0.673354 w_k1_IBM_s115
                                 + 0.783196 w_k1_INTC_s115
                                 + 0.983023 w_k1_KO_s115
                                 + 0.866885 w_k1_MAT_s115
                                 + 0.945773 w_k1_MSFT_s115
                                 + 0.877776 w_k1_PG_s115 >= 1
 z_k1_s1151_constraint:          var_1 + z_k1_s1151 + 0.82377 w_k1_ADM_s115
                                 + 0.712347 w_k1_AXP_s115
                                 + 0.652338 w_k1_DIS_s115
                                 + 0.710111 w_k1_FDX_s115
                                 + 0.680327 w_k1_IBM_s115
                                 + 0.706022 w_k1_INTC_s115
                                 + 0.695856 w_k1_KO_s115
                                 + 0.890611 w_k1_MAT_s115
                                 + 0.791672 w_k1_MSFT_s115
                                 + 0.768457 w_k1_PG_s115 >= 1
 z_k1_s1152_constraint:          var_1 + z_k1_s1152 + 0.846347 w_k1_ADM_s115
                                 + 1.3988 w_k1_AXP_s115 + 1.38062 w_k1_DIS_s115
                                 + 0.998403 w_k1_FDX_s115
                                 + 1.72398 w_k1_IBM_s115
                                 + 0.926973 w_k1_INTC_s115
                                 + 1.2934 w_k1_KO_s115 + 2.81707 w_k1_MAT_s115
                                 + 1.13902 w_k1_MSFT_s115
                                 + 1.22492 w_k1_PG_s115 >= 1
 z_k1_s1153_constraint:          var_1 + z_k1_s1153 + 1.25733 w_k1_ADM_s115
                                 + 1.13868 w_k1_AXP_s115 + 1.3549 w_k1_DIS_s115
                                 + 1.19576 w_k1_FDX_s115
                                 + 0.842056 w_k1_IBM_s115
                                 + 0.986899 w_k1_INTC_s115
                                 + 1.08902 w_k1_KO_s115 + 1.13605 w_k1_MAT_s115
                                 + 1.15686 w_k1_MSFT_s115
                                 + 1.04421 w_k1_PG_s115 >= 1
 z_k1_s1154_constraint:          var_1 + z_k1_s1154 + 0.754655 w_k1_ADM_s115
                                 + 0.865083 w_k1_AXP_s115
                                 + 0.734151 w_k1_DIS_s115
                                 + 0.792751 w_k1_FDX_s115
                                 + 1.05284 w_k1_IBM_s115
                                 + 0.720685 w_k1_INTC_s115
                                 + 0.908595 w_k1_KO_s115
                                 + 1.46128 w_k1_MAT_s115
                                 + 0.733316 w_k1_MSFT_s115
                                 + 0.825334 w_k1_PG_s115 >= 1
 z_k1_s1155_constraint:          var_1 + z_k1_s1155 + 0.88582 w_k1_ADM_s115
                                 + 0.957129 w_k1_AXP_s115
                                 + 1.28256 w_k1_DIS_s115
                                 + 0.954193 w_k1_FDX_s115
                                 + 1.43694 w_k1_IBM_s115
                                 + 0.956384 w_k1_INTC_s115
                                 + 1.2384 w_k1_KO_s115 + 1.46357 w_k1_MAT_s115
                                 + 1.07949 w_k1_MSFT_s115
                                 + 1.03205 w_k1_PG_s115 >= 1
 z_k1_s1156_constraint:          var_1 + z_k1_s1156 + 0.993468 w_k1_ADM_s115
                                 + 0.92807 w_k1_AXP_s115
                                 + 0.945417 w_k1_DIS_s115
                                 + 0.909008 w_k1_FDX_s115
                                 + 0.965692 w_k1_IBM_s115
                                 + 0.921307 w_k1_INTC_s115
                                 + 1.06362 w_k1_KO_s115 + 1.10523 w_k1_MAT_s115
                                 + 0.946439 w_k1_MSFT_s115
                                 + 0.869714 w_k1_PG_s115 >= 1
 z_k1_s1157_constraint:          var_1 + z_k1_s1157 + 1.22815 w_k1_ADM_s115
                                 + 1.36052 w_k1_AXP_s115
                                 + 1.27492 w_k1_DIS_s115
                                 + 1.17691 w_k1_FDX_s115 + 1.1552 w_k1_IBM_s115
                                 + 1.20266 w_k1_INTC_s115
                                 + 1.15974 w_k1_KO_s115 + 1.13608 w_k1_MAT_s115
                                 + 1.10123 w_k1_MSFT_s115
                                 + 1.11344 w_k1_PG_s115 >= 1
 z_k1_s1160_constraint:          var_1 + z_k1_s1160 + 1.1594 w_k1_ADM_s116
                                 + 1.19379 w_k1_AXP_s116
                                 + 1.18855 w_k1_DIS_s116
                                 + 1.08969 w_k1_FDX_s116
                                 + 1.47418 w_k1_IBM_s116
                                 + 1.22446 w_k1_INTC_s116
                                 + 1.43542 w_k1_KO_s116 + 1.5784 w_k1_MAT_s116
                                 + 1.28962 w_k1_MSFT_s116
                                 + 1.15581 w_k1_PG_s116 >= 1
 z_k1_s1161_constraint:          var_1 + z_k1_s1161 + 0.921767 w_k1_ADM_s116
                                 + 0.823611 w_k1_AXP_s116
                                 + 0.616987 w_k1_DIS_s116
                                 + 0.859587 w_k1_FDX_s116
                                 + 0.750631 w_k1_IBM_s116
                                 + 0.806656 w_k1_INTC_s116
                                 + 0.757383 w_k1_KO_s116
                                 + 0.808205 w_k1_MAT_s116
                                 + 0.864927 w_k1_MSFT_s116
                                 + 0.920756 w_k1_PG_s116 >= 1
 z_k1_s1162_constraint:          var_1 + z_k1_s1162 + 0.790085 w_k1_ADM_s116
                                 + 0.774099 w_k1_AXP_s116
                                 + 0.785373 w_k1_DIS_s116
                                 + 0.803684 w_k1_FDX_s116
                                 + 1.07119 w_k1_IBM_s116
                                 + 0.859345 w_k1_INTC_s116
                                 + 0.962199 w_k1_KO_s116
                                 + 0.898068 w_k1_MAT_s116
                                 + 0.872665 w_k1_MSFT_s116
                                 + 1.01376 w_k1_PG_s116 >= 1
 z_k1_s1163_constraint:          var_1 + z_k1_s1163 + 0.880492 w_k1_ADM_s116
                                 + 0.613733 w_k1_AXP_s116
                                 + 0.921809 w_k1_DIS_s116
                                 + 0.95908 w_k1_FDX_s116
                                 + 0.672146 w_k1_IBM_s116
                                 + 0.709501 w_k1_INTC_s116
                                 + 0.882588 w_k1_KO_s116
                                 + 0.589108 w_k1_MAT_s116
                                 + 0.999075 w_k1_MSFT_s116
                                 + 0.939814 w_k1_PG_s116 >= 1
 z_k1_s1164_constraint:          var_1 + z_k1_s1164 + 1.30315 w_k1_ADM_s116
                                 + 1.59434 w_k1_AXP_s116
                                 + 1.91591 w_k1_DIS_s116
                                 + 1.55072 w_k1_FDX_s116 + 1.3 w_k1_IBM_s116
                                 + 1.23671 w_k1_INTC_s116
                                 + 1.52252 w_k1_KO_s116 + 1.74378 w_k1_MAT_s116
                                 + 1.45554 w_k1_MSFT_s116
                                 + 1.24897 w_k1_PG_s116 >= 1
 z_k1_s1165_constraint:          var_1 + z_k1_s1165 + 0.878936 w_k1_ADM_s116
                                 + 0.642007 w_k1_AXP_s116
                                 + 0.574898 w_k1_DIS_s116
                                 + 0.762841 w_k1_FDX_s116
                                 + 0.665883 w_k1_IBM_s116
                                 + 0.655667 w_k1_INTC_s116
                                 + 0.62957 w_k1_KO_s116
                                 + 0.792289 w_k1_MAT_s116
                                 + 0.759482 w_k1_MSFT_s116
                                 + 0.807602 w_k1_PG_s116 >= 1
 z_k1_s1166_constraint:          var_1 + z_k1_s1166 + 0.994273 w_k1_ADM_s116
                                 + 1.08789 w_k1_AXP_s116
                                 + 1.03273 w_k1_DIS_s116
                                 + 1.15986 w_k1_FDX_s116
                                 + 0.798923 w_k1_IBM_s116
                                 + 1.04647 w_k1_INTC_s116
                                 + 0.928989 w_k1_KO_s116
                                 + 0.865579 w_k1_MAT_s116
                                 + 1.1734 w_k1_MSFT_s116 + 1.06099 w_k1_PG_s116
                                 >= 1
 z_k1_s1167_constraint:          var_1 + z_k1_s1167 + 1.05887 w_k1_ADM_s116
                                 + 1.21848 w_k1_AXP_s116
                                 + 1.10672 w_k1_DIS_s116
                                 + 1.21822 w_k1_FDX_s116
                                 + 0.983536 w_k1_IBM_s116
                                 + 1.25177 w_k1_INTC_s116
                                 + 1.02258 w_k1_KO_s116 + 1.12061 w_k1_MAT_s116
                                 + 1.08881 w_k1_MSFT_s116
                                 + 0.965123 w_k1_PG_s116 >= 1
 z_k1_s1170_constraint:          var_1 + z_k1_s1170 + 0.972362 w_k1_ADM_s117
                                 + 1.0528 w_k1_AXP_s117 + 1.22881 w_k1_DIS_s117
                                 + 1.0059 w_k1_FDX_s117 + 1.12235 w_k1_IBM_s117
                                 + 1.07848 w_k1_INTC_s117 + 1.0388 w_k1_KO_s117
                                 + 1.26783 w_k1_MAT_s117
                                 + 1.13072 w_k1_MSFT_s117
                                 + 0.963795 w_k1_PG_s117 >= 1
 z_k1_s1171_constraint:          var_1 + z_k1_s1171 + 1.03402 w_k1_ADM_s117
                                 + 1.0627 w_k1_AXP_s117 + 1.24151 w_k1_DIS_s117
                                 + 1.21362 w_k1_FDX_s117
                                 + 0.757299 w_k1_IBM_s117
                                 + 1.15696 w_k1_INTC_s117
                                 + 0.870865 w_k1_KO_s117
                                 + 0.852604 w_k1_MAT_s117
                                 + 1.23629 w_k1_MSFT_s117
                                 + 0.962093 w_k1_PG_s117 >= 1
 z_k1_s1172_constraint:          var_1 + z_k1_s1172 + 1.08501 w_k1_ADM_s117
                                 + 1.11781 w_k1_AXP_s117
                                 + 1.40187 w_k1_DIS_s117
                                 + 1.21319 w_k1_FDX_s117
                                 + 1.03034 w_k1_IBM_s117
                                 + 1.33577 w_k1_INTC_s117
                                 + 1.08578 w_k1_KO_s117 + 1.02285 w_k1_MAT_s117
                                 + 1.29067 w_k1_MSFT_s117 + 1.0794 w_k1_PG_s117
                                 >= 1
 z_k1_s1173_constraint:          var_1 + z_k1_s1173 + 0.911319 w_k1_ADM_s117
                                 + 0.809842 w_k1_AXP_s117
                                 + 0.800818 w_k1_DIS_s117
                                 + 0.770553 w_k1_FDX_s117
                                 + 1.31581 w_k1_IBM_s117
                                 + 0.845445 w_k1_INTC_s117
                                 + 1.06069 w_k1_KO_s117 + 1.5862 w_k1_MAT_s117
                                 + 0.927523 w_k1_MSFT_s117
                                 + 0.92447 w_k1_PG_s117 >= 1
 z_k1_s1174_constraint:          var_1 + z_k1_s1174 + 0.98764 w_k1_ADM_s117
                                 + 1.19031 w_k1_AXP_s117
                                 + 0.946917 w_k1_DIS_s117
                                 + 1.0638 w_k1_FDX_s117
                                 + 0.936806 w_k1_IBM_s117
                                 + 0.942289 w_k1_INTC_s117
                                 + 0.845667 w_k1_KO_s117
                                 + 0.936278 w_k1_MAT_s117
                                 + 0.95253 w_k1_MSFT_s117
                                 + 0.934148 w_k1_PG_s117 >= 1
 z_k1_s1175_constraint:          var_1 + z_k1_s1175 + 1.18036 w_k1_ADM_s117
                                 + 1.14849 w_k1_AXP_s117
                                 + 1.25078 w_k1_DIS_s117
                                 + 1.05515 w_k1_FDX_s117
                                 + 1.72445 w_k1_IBM_s117
                                 + 1.02104 w_k1_INTC_s117
                                 + 1.31831 w_k1_KO_s117 + 2.29237 w_k1_MAT_s117
                                 + 1.13243 w_k1_MSFT_s117 + 1.1131 w_k1_PG_s117
                                 >= 1
 z_k1_s1176_constraint:          var_1 + z_k1_s1176 + 1.07228 w_k1_ADM_s117
                                 + 1.37815 w_k1_AXP_s117 + 1.5112 w_k1_DIS_s117
                                 + 1.329 w_k1_FDX_s117 + 1.32213 w_k1_IBM_s117
                                 + 1.07473 w_k1_INTC_s117
                                 + 1.32554 w_k1_KO_s117 + 1.94644 w_k1_MAT_s117
                                 + 1.40717 w_k1_MSFT_s117
                                 + 1.13748 w_k1_PG_s117 >= 1
 z_k1_s1177_constraint:          var_1 + z_k1_s1177 + 0.577736 w_k1_ADM_s117
                                 + 0.693421 w_k1_AXP_s117
                                 + 0.7904 w_k1_DIS_s117
                                 + 0.784155 w_k1_FDX_s117
                                 + 0.849362 w_k1_IBM_s117
                                 + 0.860633 w_k1_INTC_s117
                                 + 0.856159 w_k1_KO_s117
                                 + 1.12601 w_k1_MAT_s117
                                 + 0.925197 w_k1_MSFT_s117
                                 + 0.873832 w_k1_PG_s117 >= 1
 k1_return_constraint:           0.0275799873056842 w_k1_ADM_s000
                                 + 0.0202223737720067 w_k1_ADM_s001
                                 + 0.0564181960114336 w_k1_ADM_s002
                                 + 0.0477074577124794 w_k1_ADM_s003
                                 + 0.0276531987607917 w_k1_ADM_s004
                                 + 0.0137547255465091 w_k1_ADM_s005
                                 + 0.0363997840688809 w_k1_ADM_s006
                                 + 0.0218930011710159 w_k1_ADM_s007
                                 + 0.0348846628770048 w_k1_ADM_s010
                                 + 0.0235001416115468 w_k1_ADM_s011
                                 + 0.0184851803264154 w_k1_ADM_s012
                                 + 0.0197365492724913 w_k1_ADM_s013
                                 + 0.019559469214254 w_k1_ADM_s014
                                 + 0.0142014614278813 w_k1_ADM_s015
                                 + 0.00782354587838964 w_k1_ADM_s016
                                 + 0.0215644186611709 w_k1_ADM_s017
                                 + 0.0297936582475966 w_k1_ADM_s100
                                 + 0.0318013485286603 w_k1_ADM_s101
                                 + 0.056248165633234 w_k1_ADM_s102
                                 + 0.0520407861009655 w_k1_ADM_s103
                                 + 0.0621287309569533 w_k1_ADM_s104
                                 + 0.0947283059682134 w_k1_ADM_s105
                                 + 0.0305605724655299 w_k1_ADM_s106
                                 + 0.0830513345914224 w_k1_ADM_s107
                                 + 0.0201717783998624 w_k1_ADM_s110
                                 + 0.011927233792923 w_k1_ADM_s111
                                 + 0.0143379534822817 w_k1_ADM_s112
                                 + 0.0123593803676504 w_k1_ADM_s113
                                 + 0.020391055008021 w_k1_ADM_s114
                                 + 0.035888632834542 w_k1_ADM_s115
                                 + 0.0198588775256281 w_k1_ADM_s116
                                 + 0.0435271458296095 w_k1_ADM_s117
                                 + 0.0269608064116364 w_k1_AXP_s000
                                 + 0.0207559730352514 w_k1_AXP_s001
                                 + 0.0567684858961902 w_k1_AXP_s002
                                 + 0.0445982057334286 w_k1_AXP_s003
                                 + 0.0262306169849033 w_k1_AXP_s004
                                 + 0.0121430412861481 w_k1_AXP_s005
                                 + 0.034186707856204 w_k1_AXP_s006
                                 + 0.0199521335461394 w_k1_AXP_s007
                                 + 0.0338243707964854 w_k1_AXP_s010
                                 + 0.021604681912573 w_k1_AXP_s011
                                 + 0.0198985551482478 w_k1_AXP_s012
                                 + 0.0179842192254681 w_k1_AXP_s013
                                 + 0.0218697555685605 w_k1_AXP_s014
                                 + 0.0167059858874412 w_k1_AXP_s015
                                 + 0.00817890299969201 w_k1_AXP_s016
                                 + 0.0253726224819432 w_k1_AXP_s017
                                 + 0.031629903044464 w_k1_AXP_s100
                                 + 0.0303112452340685 w_k1_AXP_s101
                                 + 0.0597362988556284 w_k1_AXP_s102
                                 + 0.0524284002712297 w_k1_AXP_s103
                                 + 0.057224654010117 w_k1_AXP_s104
                                 + 0.0935132267311924 w_k1_AXP_s105
                                 + 0.0281112514182595 w_k1_AXP_s106
                                 + 0.0759950293401245 w_k1_AXP_s107
                                 + 0.0215321790551607 w_k1_AXP_s110
                                 + 0.0120086520254945 w_k1_AXP_s111
                                 + 0.0122598425443368 w_k1_AXP_s112
                                 + 0.0146281547924977 w_k1_AXP_s113
                                 + 0.0224395042221538 w_k1_AXP_s114
                                 + 0.0361417314222536 w_k1_AXP_s115
                                 + 0.0201562749346254 w_k1_AXP_s116
                                 + 0.0473979076374485 w_k1_AXP_s117
                                 + 0.0254498352520177 w_k1_DIS_s000
                                 + 0.0199298153581014 w_k1_DIS_s001
                                 + 0.0581227879007728 w_k1_DIS_s002
                                 + 0.0462627042098743 w_k1_DIS_s003
                                 + 0.0242338060585284 w_k1_DIS_s004
                                 + 0.0125323064871094 w_k1_DIS_s005
                                 + 0.0365714647241567 w_k1_DIS_s006
                                 + 0.0216569930466883 w_k1_DIS_s007
                                 + 0.0341729538120334 w_k1_DIS_s010
                                 + 0.0229154457413315 w_k1_DIS_s011
                                 + 0.0204345749285327 w_k1_DIS_s012
                                 + 0.0170280802042945 w_k1_DIS_s013
                                 + 0.0199780515729093 w_k1_DIS_s014
                                 + 0.0146670035032716 w_k1_DIS_s015
                                 + 0.00866990405321505 w_k1_DIS_s016
                                 + 0.023381404450023 w_k1_DIS_s017
                                 + 0.0309573804404792 w_k1_DIS_s100
                                 + 0.0326729899088455 w_k1_DIS_s101
                                 + 0.0609357080198329 w_k1_DIS_s102
                                 + 0.0504788276733848 w_k1_DIS_s103
                                 + 0.0630138731551172 w_k1_DIS_s104
                                 + 0.089862237115939 w_k1_DIS_s105
                                 + 0.0295092646606854 w_k1_DIS_s106
                                 + 0.0806742015389978 w_k1_DIS_s107
                                 + 0.0223386459914083 w_k1_DIS_s110
                                 + 0.012979204637251 w_k1_DIS_s111
                                 + 0.0145378130717685 w_k1_DIS_s112
                                 + 0.0149997335659272 w_k1_DIS_s113
                                 + 0.0211368136164774 w_k1_DIS_s114
                                 + 0.0378197151260459 w_k1_DIS_s115
                                 + 0.0197743778370438 w_k1_DIS_s116
                                 + 0.0520118075000739 w_k1_DIS_s117
                                 + 0.0247932522884182 w_k1_FDX_s000
                                 + 0.0193522717807715 w_k1_FDX_s001
                                 + 0.0553769329887519 w_k1_FDX_s002
                                 + 0.0472134597713531 w_k1_FDX_s003
                                 + 0.0239602092106456 w_k1_FDX_s004
                                 + 0.0128291025660292 w_k1_FDX_s005
                                 + 0.035900958994351 w_k1_FDX_s006
                                 + 0.0201771187709016 w_k1_FDX_s007
                                 + 0.0337822285170765 w_k1_FDX_s010
                                 + 0.0237841454034177 w_k1_FDX_s011
                                 + 0.0202029711944735 w_k1_FDX_s012
                                 + 0.0172744300395682 w_k1_FDX_s013
                                 + 0.0199671319168622 w_k1_FDX_s014
                                 + 0.0151351186122089 w_k1_FDX_s015
                                 + 0.008376267731331 w_k1_FDX_s016
                                 + 0.0222666785927067 w_k1_FDX_s017
                                 + 0.0298528104271519 w_k1_FDX_s100
                                 + 0.0348254082462491 w_k1_FDX_s101
                                 + 0.0617662357575545 w_k1_FDX_s102
                                 + 0.0501015044389261 w_k1_FDX_s103
                                 + 0.0636152203686455 w_k1_FDX_s104
                                 + 0.089206306418906 w_k1_FDX_s105
                                 + 0.0283586993338931 w_k1_FDX_s106
                                 + 0.0749487147379054 w_k1_FDX_s107
                                 + 0.0228124835982207 w_k1_FDX_s110
                                 + 0.0116490103766388 w_k1_FDX_s111
                                 + 0.0140020144357121 w_k1_FDX_s112
                                 + 0.0152655449341872 w_k1_FDX_s113
                                 + 0.0209187943038083 w_k1_FDX_s114
                                 + 0.0352026585763119 w_k1_FDX_s115
                                 + 0.0207844007148297 w_k1_FDX_s116
                                 + 0.0487245330258484 w_k1_FDX_s117
                                 + 0.0248810859753812 w_k1_IBM_s000
                                 + 0.0185630444887024 w_k1_IBM_s001
                                 + 0.0570371747603782 w_k1_IBM_s002
                                 + 0.0480709526725421 w_k1_IBM_s003
                                 + 0.024684013923625 w_k1_IBM_s004
                                 + 0.0129922827841848 w_k1_IBM_s005
                                 + 0.0351187062766768 w_k1_IBM_s006
                                 + 0.0214022502969154 w_k1_IBM_s007
                                 + 0.0352164257132113 w_k1_IBM_s010
                                 + 0.0234784246816293 w_k1_IBM_s011
                                 + 0.018991314170486 w_k1_IBM_s012
                                 + 0.0172695307498916 w_k1_IBM_s013
                                 + 0.0206506833613255 w_k1_IBM_s014
                                 + 0.0156146695282099 w_k1_IBM_s015
                                 + 0.00846362982161875 w_k1_IBM_s016
                                 + 0.023384752842247 w_k1_IBM_s017
                                 + 0.0294354334664142 w_k1_IBM_s100
                                 + 0.0333623391039326 w_k1_IBM_s101
                                 + 0.0591575933037229 w_k1_IBM_s102
                                 + 0.0483288625165281 w_k1_IBM_s103
                                 + 0.0621866859276726 w_k1_IBM_s104
                                 + 0.0913200484411308 w_k1_IBM_s105
                                 + 0.027622560672335 w_k1_IBM_s106
                                 + 0.0788474631034661 w_k1_IBM_s107
                                 + 0.023387361796615 w_k1_IBM_s110
                                 + 0.0117211448086818 w_k1_IBM_s111
                                 + 0.014399576044727 w_k1_IBM_s112
                                 + 0.0149028560007521 w_k1_IBM_s113
                                 + 0.0207843021717096 w_k1_IBM_s114
                                 + 0.0361346302995731 w_k1_IBM_s115
                                 + 0.0202240186161457 w_k1_IBM_s116
                                 + 0.050456149140365 w_k1_IBM_s117
                                 + 0.0278666389872148 w_k1_INTC_s000
                                 + 0.0212589137165739 w_k1_INTC_s001
                                 + 0.0674320037261953 w_k1_INTC_s002
                                 + 0.0449186224544244 w_k1_INTC_s003
                                 + 0.0226332634296002 w_k1_INTC_s004
                                 + 0.0121979762221698 w_k1_INTC_s005
                                 + 0.0375234951543292 w_k1_INTC_s006
                                 + 0.0204949482313734 w_k1_INTC_s007
                                 + 0.0347814736015871 w_k1_INTC_s010
                                 + 0.0250133423133019 w_k1_INTC_s011
                                 + 0.0215013581573732 w_k1_INTC_s012
                                 + 0.0155254126232339 w_k1_INTC_s013
                                 + 0.0251664357873516 w_k1_INTC_s014
                                 + 0.0159590320174646 w_k1_INTC_s015
                                 + 0.00685800542021297 w_k1_INTC_s016
                                 + 0.0250956400690962 w_k1_INTC_s017
                                 + 0.0288619347622896 w_k1_INTC_s100
                                 + 0.0331958278810011 w_k1_INTC_s101
                                 + 0.058649133473032 w_k1_INTC_s102
                                 + 0.0522220120321481 w_k1_INTC_s103
                                 + 0.0563359290382001 w_k1_INTC_s104
                                 + 0.0908880761990409 w_k1_INTC_s105
                                 + 0.025626421404573 w_k1_INTC_s106
                                 + 0.0784451205656345 w_k1_INTC_s107
                                 + 0.0213496351061623 w_k1_INTC_s110
                                 + 0.0107465845083357 w_k1_INTC_s111
                                 + 0.014015761001059 w_k1_INTC_s112
                                 + 0.0149124774074624 w_k1_INTC_s113
                                 + 0.0220819159098814 w_k1_INTC_s114
                                 + 0.032706343233592 w_k1_INTC_s115
                                 + 0.0203186150035806 w_k1_INTC_s116
                                 + 0.0490355753556503 w_k1_INTC_s117
                                 + 0.0232692620986283 w_k1_KO_s000
                                 + 0.0190693898452813 w_k1_KO_s001
                                 + 0.053411847398138 w_k1_KO_s002
                                 + 0.0474439177072704 w_k1_KO_s003
                                 + 0.0270252603314223 w_k1_KO_s004
                                 + 0.0121731624557246 w_k1_KO_s005
                                 + 0.035600141934169 w_k1_KO_s006
                                 + 0.0220161325712764 w_k1_KO_s007
                                 + 0.0389433775047199 w_k1_KO_s010
                                 + 0.0232080369327915 w_k1_KO_s011
                                 + 0.0177056673886809 w_k1_KO_s012
                                 + 0.0177504855093018 w_k1_KO_s013
                                 + 0.0192722298174271 w_k1_KO_s014
                                 + 0.0157155076576144 w_k1_KO_s015
                                 + 0.00827691276877798 w_k1_KO_s016
                                 + 0.0249037181392546 w_k1_KO_s017
                                 + 0.0304823223656012 w_k1_KO_s100
                                 + 0.0314939322909665 w_k1_KO_s101
                                 + 0.0591260998375822 w_k1_KO_s102
                                 + 0.0496792375552033 w_k1_KO_s103
                                 + 0.0667367746048084 w_k1_KO_s104
                                 + 0.0952311425291308 w_k1_KO_s105
                                 + 0.0285290096972995 w_k1_KO_s106
                                 + 0.0708624399458903 w_k1_KO_s107
                                 + 0.0222275353582511 w_k1_KO_s110
                                 + 0.0123191813030951 w_k1_KO_s111
                                 + 0.0150139930379735 w_k1_KO_s112
                                 + 0.0152927599258589 w_k1_KO_s113
                                 + 0.0219101835208317 w_k1_KO_s114
                                 + 0.0373625502242713 w_k1_KO_s115
                                 + 0.0206424233726271 w_k1_KO_s116
                                 + 0.0479888901440101 w_k1_KO_s117
                                 + 0.022189561445074 w_k1_MAT_s000
                                 + 0.0207682462950114 w_k1_MAT_s001
                                 + 0.0463903129887905 w_k1_MAT_s002
                                 + 0.0436967036710097 w_k1_MAT_s003
                                 + 0.0242145043413916 w_k1_MAT_s004
                                 + 0.0128539014398407 w_k1_MAT_s005
                                 + 0.0429967933893773 w_k1_MAT_s006
                                 + 0.0172852883587401 w_k1_MAT_s007
                                 + 0.0399094074563427 w_k1_MAT_s010
                                 + 0.0225270021198681 w_k1_MAT_s011
                                 + 0.0178921964445531 w_k1_MAT_s012
                                 + 0.0167103790470832 w_k1_MAT_s013
                                 + 0.0216396104940466 w_k1_MAT_s014
                                 + 0.0174139630960312 w_k1_MAT_s015
                                 + 0.0108324778611917 w_k1_MAT_s016
                                 + 0.0213770491665364 w_k1_MAT_s017
                                 + 0.0287147386300986 w_k1_MAT_s100
                                 + 0.0347390043014865 w_k1_MAT_s101
                                 + 0.0536690104707928 w_k1_MAT_s102
                                 + 0.055065499760871 w_k1_MAT_s103
                                 + 0.0620998133009444 w_k1_MAT_s104
                                 + 0.102824922104766 w_k1_MAT_s105
                                 + 0.0316492609371512 w_k1_MAT_s106
                                 + 0.0702095576513845 w_k1_MAT_s107
                                 + 0.0222177094420846 w_k1_MAT_s110
                                 + 0.00970437482666749 w_k1_MAT_s111
                                 + 0.0148788786847468 w_k1_MAT_s112
                                 + 0.0158106121312504 w_k1_MAT_s113
                                 + 0.0186028403264793 w_k1_MAT_s114
                                 + 0.0438745865038083 w_k1_MAT_s115
                                 + 0.0212726834222643 w_k1_MAT_s116
                                 + 0.0616482439186736 w_k1_MAT_s117
                                 + 0.0261246871744523 w_k1_MSFT_s000
                                 + 0.021425776957387 w_k1_MSFT_s001
                                 + 0.0609573743863278 w_k1_MSFT_s002
                                 + 0.0422853439014715 w_k1_MSFT_s003
                                 + 0.023648240343671 w_k1_MSFT_s004
                                 + 0.0126022354879422 w_k1_MSFT_s005
                                 + 0.0393813884139236 w_k1_MSFT_s006
                                 + 0.0210044007831296 w_k1_MSFT_s007
                                 + 0.03229711376358 w_k1_MSFT_s010
                                 + 0.0227964301408191 w_k1_MSFT_s011
                                 + 0.0182482139894505 w_k1_MSFT_s012
                                 + 0.0183707179632052 w_k1_MSFT_s013
                                 + 0.0197285684998622 w_k1_MSFT_s014
                                 + 0.014771129356747 w_k1_MSFT_s015
                                 + 0.00864782008885892 w_k1_MSFT_s016
                                 + 0.0256522634169363 w_k1_MSFT_s017
                                 + 0.0318084860658818 w_k1_MSFT_s100
                                 + 0.0325257285476818 w_k1_MSFT_s101
                                 + 0.0600148112623331 w_k1_MSFT_s102
                                 + 0.0509290997689711 w_k1_MSFT_s103
                                 + 0.0578373889729057 w_k1_MSFT_s104
                                 + 0.0882239950385015 w_k1_MSFT_s105
                                 + 0.0280599551567441 w_k1_MSFT_s106
                                 + 0.0753931374098751 w_k1_MSFT_s107
                                 + 0.0214168893710406 w_k1_MSFT_s110
                                 + 0.0114793806791876 w_k1_MSFT_s111
                                 + 0.0142268992169955 w_k1_MSFT_s112
                                 + 0.0141389830210558 w_k1_MSFT_s113
                                 + 0.0227941165647872 w_k1_MSFT_s114
                                 + 0.0348983893514192 w_k1_MSFT_s115
                                 + 0.0213476332522506 w_k1_MSFT_s116
                                 + 0.0527290090519148 w_k1_MSFT_s117
                                 + 0.0255120277250423 w_k1_PG_s000
                                 + 0.0193026113051565 w_k1_PG_s001
                                 + 0.0539522599283214 w_k1_PG_s002
                                 + 0.0491718432362578 w_k1_PG_s003
                                 + 0.0241443580034032 w_k1_PG_s004
                                 + 0.0109967317387303 w_k1_PG_s005
                                 + 0.0368054907788411 w_k1_PG_s006
                                 + 0.0192697208412923 w_k1_PG_s007
                                 + 0.0343795548540672 w_k1_PG_s010
                                 + 0.0239759640772542 w_k1_PG_s011
                                 + 0.0200027427688005 w_k1_PG_s012
                                 + 0.0163954722013604 w_k1_PG_s013
                                 + 0.0202569191290484 w_k1_PG_s014
                                 + 0.0141564621625817 w_k1_PG_s015
                                 + 0.00803569310136764 w_k1_PG_s016
                                 + 0.0239382459756151 w_k1_PG_s017
                                 + 0.0316748196680894 w_k1_PG_s100
                                 + 0.0335593421698546 w_k1_PG_s101
                                 + 0.0548407461338395 w_k1_PG_s102
                                 + 0.0455580068137451 w_k1_PG_s103
                                 + 0.0605096880891196 w_k1_PG_s104
                                 + 0.0916114157463221 w_k1_PG_s105
                                 + 0.0253097373326858 w_k1_PG_s106
                                 + 0.0705648788887018 w_k1_PG_s107
                                 + 0.0221772186354976 w_k1_PG_s110
                                 + 0.0111270943466065 w_k1_PG_s111
                                 + 0.0140520344973619 w_k1_PG_s112
                                 + 0.0151816376527216 w_k1_PG_s113
                                 + 0.0211090821315042 w_k1_PG_s114
                                 + 0.033962138836677 w_k1_PG_s115
                                 + 0.020545241169028 w_k1_PG_s116
                                 + 0.0469146968807976 w_k1_PG_s117 >= 1.1
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
