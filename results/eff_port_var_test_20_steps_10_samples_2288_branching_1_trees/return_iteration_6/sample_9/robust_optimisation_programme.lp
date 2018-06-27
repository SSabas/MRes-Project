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
 w_k1_ADM_s0_return_constraint:  0.977145 w_0_ADM - w_k1_ADM_s0
                                 + 0.99 b_k1_ADM_s0 - 1.01 s_k1_ADM_s0  = 0
 w_k1_ADM_s1_return_constraint:  0.97062 w_0_ADM - w_k1_ADM_s1
                                 + 0.99 b_k1_ADM_s1 - 1.01 s_k1_ADM_s1  = 0
 w_k1_ADM_s00_return_constraint: 1.43645 w_k1_ADM_s0 - w_k1_ADM_s00
                                 + 0.99 b_k1_ADM_s00 - 1.01 s_k1_ADM_s00  = 0
 w_k1_ADM_s01_return_constraint: 1.02024 w_k1_ADM_s0 - w_k1_ADM_s01
                                 + 0.99 b_k1_ADM_s01 - 1.01 s_k1_ADM_s01  = 0
 w_k1_ADM_s10_return_constraint: 1.26289 w_k1_ADM_s1 - w_k1_ADM_s10
                                 + 0.99 b_k1_ADM_s10 - 1.01 s_k1_ADM_s10  = 0
 w_k1_ADM_s11_return_constraint: 0.91621 w_k1_ADM_s1 - w_k1_ADM_s11
                                 + 0.99 b_k1_ADM_s11 - 1.01 s_k1_ADM_s11  = 0
 w_k1_ADM_s000_return_constraint: 0.942229 w_k1_ADM_s00 - w_k1_ADM_s000
                                 + 0.99 b_k1_ADM_s000 - 1.01 s_k1_ADM_s000  = 0
 w_k1_ADM_s001_return_constraint: 1.2456 w_k1_ADM_s00 - w_k1_ADM_s001
                                 + 0.99 b_k1_ADM_s001 - 1.01 s_k1_ADM_s001  = 0
 w_k1_ADM_s002_return_constraint: 0.647945 w_k1_ADM_s00 - w_k1_ADM_s002
                                 + 0.99 b_k1_ADM_s002 - 1.01 s_k1_ADM_s002  = 0
 w_k1_ADM_s003_return_constraint: 1.07212 w_k1_ADM_s00 - w_k1_ADM_s003
                                 + 0.99 b_k1_ADM_s003 - 1.01 s_k1_ADM_s003  = 0
 w_k1_ADM_s004_return_constraint: 0.781559 w_k1_ADM_s00 - w_k1_ADM_s004
                                 + 0.99 b_k1_ADM_s004 - 1.01 s_k1_ADM_s004  = 0
 w_k1_ADM_s005_return_constraint: 1.00129 w_k1_ADM_s00 - w_k1_ADM_s005
                                 + 0.99 b_k1_ADM_s005 - 1.01 s_k1_ADM_s005  = 0
 w_k1_ADM_s006_return_constraint: 1.06652 w_k1_ADM_s00 - w_k1_ADM_s006
                                 + 0.99 b_k1_ADM_s006 - 1.01 s_k1_ADM_s006  = 0
 w_k1_ADM_s007_return_constraint: 0.77196 w_k1_ADM_s00 - w_k1_ADM_s007
                                 + 0.99 b_k1_ADM_s007 - 1.01 s_k1_ADM_s007  = 0
 w_k1_ADM_s010_return_constraint: 1.21627 w_k1_ADM_s01 - w_k1_ADM_s010
                                 + 0.99 b_k1_ADM_s010 - 1.01 s_k1_ADM_s010  = 0
 w_k1_ADM_s011_return_constraint: 1.02805 w_k1_ADM_s01 - w_k1_ADM_s011
                                 + 0.99 b_k1_ADM_s011 - 1.01 s_k1_ADM_s011  = 0
 w_k1_ADM_s012_return_constraint: 0.623696 w_k1_ADM_s01 - w_k1_ADM_s012
                                 + 0.99 b_k1_ADM_s012 - 1.01 s_k1_ADM_s012  = 0
 w_k1_ADM_s013_return_constraint: 1.11563 w_k1_ADM_s01 - w_k1_ADM_s013
                                 + 0.99 b_k1_ADM_s013 - 1.01 s_k1_ADM_s013  = 0
 w_k1_ADM_s014_return_constraint: 1.37103 w_k1_ADM_s01 - w_k1_ADM_s014
                                 + 0.99 b_k1_ADM_s014 - 1.01 s_k1_ADM_s014  = 0
 w_k1_ADM_s015_return_constraint: 0.876961 w_k1_ADM_s01 - w_k1_ADM_s015
                                 + 0.99 b_k1_ADM_s015 - 1.01 s_k1_ADM_s015  = 0
 w_k1_ADM_s016_return_constraint: 1.29944 w_k1_ADM_s01 - w_k1_ADM_s016
                                 + 0.99 b_k1_ADM_s016 - 1.01 s_k1_ADM_s016  = 0
 w_k1_ADM_s017_return_constraint: 0.99497 w_k1_ADM_s01 - w_k1_ADM_s017
                                 + 0.99 b_k1_ADM_s017 - 1.01 s_k1_ADM_s017  = 0
 w_k1_ADM_s100_return_constraint: 0.894989 w_k1_ADM_s10 - w_k1_ADM_s100
                                 + 0.99 b_k1_ADM_s100 - 1.01 s_k1_ADM_s100  = 0
 w_k1_ADM_s101_return_constraint: 0.820129 w_k1_ADM_s10 - w_k1_ADM_s101
                                 + 0.99 b_k1_ADM_s101 - 1.01 s_k1_ADM_s101  = 0
 w_k1_ADM_s102_return_constraint: 0.881146 w_k1_ADM_s10 - w_k1_ADM_s102
                                 + 0.99 b_k1_ADM_s102 - 1.01 s_k1_ADM_s102  = 0
 w_k1_ADM_s103_return_constraint: 1.40471 w_k1_ADM_s10 - w_k1_ADM_s103
                                 + 0.99 b_k1_ADM_s103 - 1.01 s_k1_ADM_s103  = 0
 w_k1_ADM_s104_return_constraint: 0.857676 w_k1_ADM_s10 - w_k1_ADM_s104
                                 + 0.99 b_k1_ADM_s104 - 1.01 s_k1_ADM_s104  = 0
 w_k1_ADM_s105_return_constraint: 0.997164 w_k1_ADM_s10 - w_k1_ADM_s105
                                 + 0.99 b_k1_ADM_s105 - 1.01 s_k1_ADM_s105  = 0
 w_k1_ADM_s106_return_constraint: 1.02155 w_k1_ADM_s10 - w_k1_ADM_s106
                                 + 0.99 b_k1_ADM_s106 - 1.01 s_k1_ADM_s106  = 0
 w_k1_ADM_s107_return_constraint: 0.815811 w_k1_ADM_s10 - w_k1_ADM_s107
                                 + 0.99 b_k1_ADM_s107 - 1.01 s_k1_ADM_s107  = 0
 w_k1_ADM_s110_return_constraint: 1.2818 w_k1_ADM_s11 - w_k1_ADM_s110
                                 + 0.99 b_k1_ADM_s110 - 1.01 s_k1_ADM_s110  = 0
 w_k1_ADM_s111_return_constraint: 0.953703 w_k1_ADM_s11 - w_k1_ADM_s111
                                 + 0.99 b_k1_ADM_s111 - 1.01 s_k1_ADM_s111  = 0
 w_k1_ADM_s112_return_constraint: 0.962085 w_k1_ADM_s11 - w_k1_ADM_s112
                                 + 0.99 b_k1_ADM_s112 - 1.01 s_k1_ADM_s112  = 0
 w_k1_ADM_s113_return_constraint: 0.931571 w_k1_ADM_s11 - w_k1_ADM_s113
                                 + 0.99 b_k1_ADM_s113 - 1.01 s_k1_ADM_s113  = 0
 w_k1_ADM_s114_return_constraint: 0.976482 w_k1_ADM_s11 - w_k1_ADM_s114
                                 + 0.99 b_k1_ADM_s114 - 1.01 s_k1_ADM_s114  = 0
 w_k1_ADM_s115_return_constraint: 1.07718 w_k1_ADM_s11 - w_k1_ADM_s115
                                 + 0.99 b_k1_ADM_s115 - 1.01 s_k1_ADM_s115  = 0
 w_k1_ADM_s116_return_constraint: 1.21975 w_k1_ADM_s11 - w_k1_ADM_s116
                                 + 0.99 b_k1_ADM_s116 - 1.01 s_k1_ADM_s116  = 0
 w_k1_ADM_s117_return_constraint: 0.734748 w_k1_ADM_s11 - w_k1_ADM_s117
                                 + 0.99 b_k1_ADM_s117 - 1.01 s_k1_ADM_s117  = 0
 w_k1_AXP_s0_return_constraint:  0.999897 w_0_AXP - w_k1_AXP_s0
                                 + 0.99 b_k1_AXP_s0 - 1.01 s_k1_AXP_s0  = 0
 w_k1_AXP_s1_return_constraint:  1.07905 w_0_AXP - w_k1_AXP_s1
                                 + 0.99 b_k1_AXP_s1 - 1.01 s_k1_AXP_s1  = 0
 w_k1_AXP_s00_return_constraint: 1.39796 w_k1_AXP_s0 - w_k1_AXP_s00
                                 + 0.99 b_k1_AXP_s00 - 1.01 s_k1_AXP_s00  = 0
 w_k1_AXP_s01_return_constraint: 0.839163 w_k1_AXP_s0 - w_k1_AXP_s01
                                 + 0.99 b_k1_AXP_s01 - 1.01 s_k1_AXP_s01  = 0
 w_k1_AXP_s10_return_constraint: 1.25459 w_k1_AXP_s1 - w_k1_AXP_s10
                                 + 0.99 b_k1_AXP_s10 - 1.01 s_k1_AXP_s10  = 0
 w_k1_AXP_s11_return_constraint: 0.81591 w_k1_AXP_s1 - w_k1_AXP_s11
                                 + 0.99 b_k1_AXP_s11 - 1.01 s_k1_AXP_s11  = 0
 w_k1_AXP_s000_return_constraint: 1.04529 w_k1_AXP_s00 - w_k1_AXP_s000
                                 + 0.99 b_k1_AXP_s000 - 1.01 s_k1_AXP_s000  = 0
 w_k1_AXP_s001_return_constraint: 1.5212 w_k1_AXP_s00 - w_k1_AXP_s001
                                 + 0.99 b_k1_AXP_s001 - 1.01 s_k1_AXP_s001  = 0
 w_k1_AXP_s002_return_constraint: 0.795728 w_k1_AXP_s00 - w_k1_AXP_s002
                                 + 0.99 b_k1_AXP_s002 - 1.01 s_k1_AXP_s002  = 0
 w_k1_AXP_s003_return_constraint: 1.21638 w_k1_AXP_s00 - w_k1_AXP_s003
                                 + 0.99 b_k1_AXP_s003 - 1.01 s_k1_AXP_s003  = 0
 w_k1_AXP_s004_return_constraint: 0.98348 w_k1_AXP_s00 - w_k1_AXP_s004
                                 + 0.99 b_k1_AXP_s004 - 1.01 s_k1_AXP_s004  = 0
 w_k1_AXP_s005_return_constraint: 1.13148 w_k1_AXP_s00 - w_k1_AXP_s005
                                 + 0.99 b_k1_AXP_s005 - 1.01 s_k1_AXP_s005  = 0
 w_k1_AXP_s006_return_constraint: 1.37998 w_k1_AXP_s00 - w_k1_AXP_s006
                                 + 0.99 b_k1_AXP_s006 - 1.01 s_k1_AXP_s006  = 0
 w_k1_AXP_s007_return_constraint: 0.702823 w_k1_AXP_s00 - w_k1_AXP_s007
                                 + 0.99 b_k1_AXP_s007 - 1.01 s_k1_AXP_s007  = 0
 w_k1_AXP_s010_return_constraint: 1.8147 w_k1_AXP_s01 - w_k1_AXP_s010
                                 + 0.99 b_k1_AXP_s010 - 1.01 s_k1_AXP_s010  = 0
 w_k1_AXP_s011_return_constraint: 1.21531 w_k1_AXP_s01 - w_k1_AXP_s011
                                 + 0.99 b_k1_AXP_s011 - 1.01 s_k1_AXP_s011  = 0
 w_k1_AXP_s012_return_constraint: 0.591798 w_k1_AXP_s01 - w_k1_AXP_s012
                                 + 0.99 b_k1_AXP_s012 - 1.01 s_k1_AXP_s012  = 0
 w_k1_AXP_s013_return_constraint: 1.27534 w_k1_AXP_s01 - w_k1_AXP_s013
                                 + 0.99 b_k1_AXP_s013 - 1.01 s_k1_AXP_s013  = 0
 w_k1_AXP_s014_return_constraint: 1.42927 w_k1_AXP_s01 - w_k1_AXP_s014
                                 + 0.99 b_k1_AXP_s014 - 1.01 s_k1_AXP_s014  = 0
 w_k1_AXP_s015_return_constraint: 0.932097 w_k1_AXP_s01 - w_k1_AXP_s015
                                 + 0.99 b_k1_AXP_s015 - 1.01 s_k1_AXP_s015  = 0
 w_k1_AXP_s016_return_constraint: 1.58035 w_k1_AXP_s01 - w_k1_AXP_s016
                                 + 0.99 b_k1_AXP_s016 - 1.01 s_k1_AXP_s016  = 0
 w_k1_AXP_s017_return_constraint: 1.02738 w_k1_AXP_s01 - w_k1_AXP_s017
                                 + 0.99 b_k1_AXP_s017 - 1.01 s_k1_AXP_s017  = 0
 w_k1_AXP_s100_return_constraint: 0.597685 w_k1_AXP_s10 - w_k1_AXP_s100
                                 + 0.99 b_k1_AXP_s100 - 1.01 s_k1_AXP_s100  = 0
 w_k1_AXP_s101_return_constraint: 0.795579 w_k1_AXP_s10 - w_k1_AXP_s101
                                 + 0.99 b_k1_AXP_s101 - 1.01 s_k1_AXP_s101  = 0
 w_k1_AXP_s102_return_constraint: 1.0226 w_k1_AXP_s10 - w_k1_AXP_s102
                                 + 0.99 b_k1_AXP_s102 - 1.01 s_k1_AXP_s102  = 0
 w_k1_AXP_s103_return_constraint: 1.36982 w_k1_AXP_s10 - w_k1_AXP_s103
                                 + 0.99 b_k1_AXP_s103 - 1.01 s_k1_AXP_s103  = 0
 w_k1_AXP_s104_return_constraint: 0.792669 w_k1_AXP_s10 - w_k1_AXP_s104
                                 + 0.99 b_k1_AXP_s104 - 1.01 s_k1_AXP_s104  = 0
 w_k1_AXP_s105_return_constraint: 1.41065 w_k1_AXP_s10 - w_k1_AXP_s105
                                 + 0.99 b_k1_AXP_s105 - 1.01 s_k1_AXP_s105  = 0
 w_k1_AXP_s106_return_constraint: 1.00022 w_k1_AXP_s10 - w_k1_AXP_s106
                                 + 0.99 b_k1_AXP_s106 - 1.01 s_k1_AXP_s106  = 0
 w_k1_AXP_s107_return_constraint: 0.814885 w_k1_AXP_s10 - w_k1_AXP_s107
                                 + 0.99 b_k1_AXP_s107 - 1.01 s_k1_AXP_s107  = 0
 w_k1_AXP_s110_return_constraint: 1.68162 w_k1_AXP_s11 - w_k1_AXP_s110
                                 + 0.99 b_k1_AXP_s110 - 1.01 s_k1_AXP_s110  = 0
 w_k1_AXP_s111_return_constraint: 0.878489 w_k1_AXP_s11 - w_k1_AXP_s111
                                 + 0.99 b_k1_AXP_s111 - 1.01 s_k1_AXP_s111  = 0
 w_k1_AXP_s112_return_constraint: 0.821016 w_k1_AXP_s11 - w_k1_AXP_s112
                                 + 0.99 b_k1_AXP_s112 - 1.01 s_k1_AXP_s112  = 0
 w_k1_AXP_s113_return_constraint: 0.782143 w_k1_AXP_s11 - w_k1_AXP_s113
                                 + 0.99 b_k1_AXP_s113 - 1.01 s_k1_AXP_s113  = 0
 w_k1_AXP_s114_return_constraint: 1.15253 w_k1_AXP_s11 - w_k1_AXP_s114
                                 + 0.99 b_k1_AXP_s114 - 1.01 s_k1_AXP_s114  = 0
 w_k1_AXP_s115_return_constraint: 1.07409 w_k1_AXP_s11 - w_k1_AXP_s115
                                 + 0.99 b_k1_AXP_s115 - 1.01 s_k1_AXP_s115  = 0
 w_k1_AXP_s116_return_constraint: 1.22772 w_k1_AXP_s11 - w_k1_AXP_s116
                                 + 0.99 b_k1_AXP_s116 - 1.01 s_k1_AXP_s116  = 0
 w_k1_AXP_s117_return_constraint: 0.682989 w_k1_AXP_s11 - w_k1_AXP_s117
                                 + 0.99 b_k1_AXP_s117 - 1.01 s_k1_AXP_s117  = 0
 w_k1_DIS_s0_return_constraint:  0.995929 w_0_DIS - w_k1_DIS_s0
                                 + 0.99 b_k1_DIS_s0 - 1.01 s_k1_DIS_s0  = 0
 w_k1_DIS_s1_return_constraint:  1.11194 w_0_DIS - w_k1_DIS_s1
                                 + 0.99 b_k1_DIS_s1 - 1.01 s_k1_DIS_s1  = 0
 w_k1_DIS_s00_return_constraint: 1.48948 w_k1_DIS_s0 - w_k1_DIS_s00
                                 + 0.99 b_k1_DIS_s00 - 1.01 s_k1_DIS_s00  = 0
 w_k1_DIS_s01_return_constraint: 0.892606 w_k1_DIS_s0 - w_k1_DIS_s01
                                 + 0.99 b_k1_DIS_s01 - 1.01 s_k1_DIS_s01  = 0
 w_k1_DIS_s10_return_constraint: 1.38484 w_k1_DIS_s1 - w_k1_DIS_s10
                                 + 0.99 b_k1_DIS_s10 - 1.01 s_k1_DIS_s10  = 0
 w_k1_DIS_s11_return_constraint: 0.744384 w_k1_DIS_s1 - w_k1_DIS_s11
                                 + 0.99 b_k1_DIS_s11 - 1.01 s_k1_DIS_s11  = 0
 w_k1_DIS_s000_return_constraint: 0.866229 w_k1_DIS_s00 - w_k1_DIS_s000
                                 + 0.99 b_k1_DIS_s000 - 1.01 s_k1_DIS_s000  = 0
 w_k1_DIS_s001_return_constraint: 1.83405 w_k1_DIS_s00 - w_k1_DIS_s001
                                 + 0.99 b_k1_DIS_s001 - 1.01 s_k1_DIS_s001  = 0
 w_k1_DIS_s002_return_constraint: 0.727389 w_k1_DIS_s00 - w_k1_DIS_s002
                                 + 0.99 b_k1_DIS_s002 - 1.01 s_k1_DIS_s002  = 0
 w_k1_DIS_s003_return_constraint: 1.08591 w_k1_DIS_s00 - w_k1_DIS_s003
                                 + 0.99 b_k1_DIS_s003 - 1.01 s_k1_DIS_s003  = 0
 w_k1_DIS_s004_return_constraint: 0.878117 w_k1_DIS_s00 - w_k1_DIS_s004
                                 + 0.99 b_k1_DIS_s004 - 1.01 s_k1_DIS_s004  = 0
 w_k1_DIS_s005_return_constraint: 1.27093 w_k1_DIS_s00 - w_k1_DIS_s005
                                 + 0.99 b_k1_DIS_s005 - 1.01 s_k1_DIS_s005  = 0
 w_k1_DIS_s006_return_constraint: 1.41159 w_k1_DIS_s00 - w_k1_DIS_s006
                                 + 0.99 b_k1_DIS_s006 - 1.01 s_k1_DIS_s006  = 0
 w_k1_DIS_s007_return_constraint: 0.681023 w_k1_DIS_s00 - w_k1_DIS_s007
                                 + 0.99 b_k1_DIS_s007 - 1.01 s_k1_DIS_s007  = 0
 w_k1_DIS_s010_return_constraint: 1.6567 w_k1_DIS_s01 - w_k1_DIS_s010
                                 + 0.99 b_k1_DIS_s010 - 1.01 s_k1_DIS_s010  = 0
 w_k1_DIS_s011_return_constraint: 1.21364 w_k1_DIS_s01 - w_k1_DIS_s011
                                 + 0.99 b_k1_DIS_s011 - 1.01 s_k1_DIS_s011  = 0
 w_k1_DIS_s012_return_constraint: 0.48307 w_k1_DIS_s01 - w_k1_DIS_s012
                                 + 0.99 b_k1_DIS_s012 - 1.01 s_k1_DIS_s012  = 0
 w_k1_DIS_s013_return_constraint: 0.980786 w_k1_DIS_s01 - w_k1_DIS_s013
                                 + 0.99 b_k1_DIS_s013 - 1.01 s_k1_DIS_s013  = 0
 w_k1_DIS_s014_return_constraint: 1.39494 w_k1_DIS_s01 - w_k1_DIS_s014
                                 + 0.99 b_k1_DIS_s014 - 1.01 s_k1_DIS_s014  = 0
 w_k1_DIS_s015_return_constraint: 0.78949 w_k1_DIS_s01 - w_k1_DIS_s015
                                 + 0.99 b_k1_DIS_s015 - 1.01 s_k1_DIS_s015  = 0
 w_k1_DIS_s016_return_constraint: 1.83918 w_k1_DIS_s01 - w_k1_DIS_s016
                                 + 0.99 b_k1_DIS_s016 - 1.01 s_k1_DIS_s016  = 0
 w_k1_DIS_s017_return_constraint: 1.05537 w_k1_DIS_s01 - w_k1_DIS_s017
                                 + 0.99 b_k1_DIS_s017 - 1.01 s_k1_DIS_s017  = 0
 w_k1_DIS_s100_return_constraint: 0.803922 w_k1_DIS_s10 - w_k1_DIS_s100
                                 + 0.99 b_k1_DIS_s100 - 1.01 s_k1_DIS_s100  = 0
 w_k1_DIS_s101_return_constraint: 0.667857 w_k1_DIS_s10 - w_k1_DIS_s101
                                 + 0.99 b_k1_DIS_s101 - 1.01 s_k1_DIS_s101  = 0
 w_k1_DIS_s102_return_constraint: 0.907203 w_k1_DIS_s10 - w_k1_DIS_s102
                                 + 0.99 b_k1_DIS_s102 - 1.01 s_k1_DIS_s102  = 0
 w_k1_DIS_s103_return_constraint: 1.47047 w_k1_DIS_s10 - w_k1_DIS_s103
                                 + 0.99 b_k1_DIS_s103 - 1.01 s_k1_DIS_s103  = 0
 w_k1_DIS_s104_return_constraint: 0.691031 w_k1_DIS_s10 - w_k1_DIS_s104
                                 + 0.99 b_k1_DIS_s104 - 1.01 s_k1_DIS_s104  = 0
 w_k1_DIS_s105_return_constraint: 1.38195 w_k1_DIS_s10 - w_k1_DIS_s105
                                 + 0.99 b_k1_DIS_s105 - 1.01 s_k1_DIS_s105  = 0
 w_k1_DIS_s106_return_constraint: 1.12695 w_k1_DIS_s10 - w_k1_DIS_s106
                                 + 0.99 b_k1_DIS_s106 - 1.01 s_k1_DIS_s106  = 0
 w_k1_DIS_s107_return_constraint: 0.741072 w_k1_DIS_s10 - w_k1_DIS_s107
                                 + 0.99 b_k1_DIS_s107 - 1.01 s_k1_DIS_s107  = 0
 w_k1_DIS_s110_return_constraint: 1.89899 w_k1_DIS_s11 - w_k1_DIS_s110
                                 + 0.99 b_k1_DIS_s110 - 1.01 s_k1_DIS_s110  = 0
 w_k1_DIS_s111_return_constraint: 0.816835 w_k1_DIS_s11 - w_k1_DIS_s111
                                 + 0.99 b_k1_DIS_s111 - 1.01 s_k1_DIS_s111  = 0
 w_k1_DIS_s112_return_constraint: 0.819251 w_k1_DIS_s11 - w_k1_DIS_s112
                                 + 0.99 b_k1_DIS_s112 - 1.01 s_k1_DIS_s112  = 0
 w_k1_DIS_s113_return_constraint: 0.782246 w_k1_DIS_s11 - w_k1_DIS_s113
                                 + 0.99 b_k1_DIS_s113 - 1.01 s_k1_DIS_s113  = 0
 w_k1_DIS_s114_return_constraint: 1.12176 w_k1_DIS_s11 - w_k1_DIS_s114
                                 + 0.99 b_k1_DIS_s114 - 1.01 s_k1_DIS_s114  = 0
 w_k1_DIS_s115_return_constraint: 1.02018 w_k1_DIS_s11 - w_k1_DIS_s115
                                 + 0.99 b_k1_DIS_s115 - 1.01 s_k1_DIS_s115  = 0
 w_k1_DIS_s116_return_constraint: 1.32732 w_k1_DIS_s11 - w_k1_DIS_s116
                                 + 0.99 b_k1_DIS_s116 - 1.01 s_k1_DIS_s116  = 0
 w_k1_DIS_s117_return_constraint: 0.668563 w_k1_DIS_s11 - w_k1_DIS_s117
                                 + 0.99 b_k1_DIS_s117 - 1.01 s_k1_DIS_s117  = 0
 w_k1_FDX_s0_return_constraint:  1.02493 w_0_FDX - w_k1_FDX_s0
                                 + 0.99 b_k1_FDX_s0 - 1.01 s_k1_FDX_s0  = 0
 w_k1_FDX_s1_return_constraint:  0.98855 w_0_FDX - w_k1_FDX_s1
                                 + 0.99 b_k1_FDX_s1 - 1.01 s_k1_FDX_s1  = 0
 w_k1_FDX_s00_return_constraint: 1.26941 w_k1_FDX_s0 - w_k1_FDX_s00
                                 + 0.99 b_k1_FDX_s00 - 1.01 s_k1_FDX_s00  = 0
 w_k1_FDX_s01_return_constraint: 0.948325 w_k1_FDX_s0 - w_k1_FDX_s01
                                 + 0.99 b_k1_FDX_s01 - 1.01 s_k1_FDX_s01  = 0
 w_k1_FDX_s10_return_constraint: 1.2205 w_k1_FDX_s1 - w_k1_FDX_s10
                                 + 0.99 b_k1_FDX_s10 - 1.01 s_k1_FDX_s10  = 0
 w_k1_FDX_s11_return_constraint: 0.856414 w_k1_FDX_s1 - w_k1_FDX_s11
                                 + 0.99 b_k1_FDX_s11 - 1.01 s_k1_FDX_s11  = 0
 w_k1_FDX_s000_return_constraint: 0.945408 w_k1_FDX_s00 - w_k1_FDX_s000
                                 + 0.99 b_k1_FDX_s000 - 1.01 s_k1_FDX_s000  = 0
 w_k1_FDX_s001_return_constraint: 1.45949 w_k1_FDX_s00 - w_k1_FDX_s001
                                 + 0.99 b_k1_FDX_s001 - 1.01 s_k1_FDX_s001  = 0
 w_k1_FDX_s002_return_constraint: 0.71556 w_k1_FDX_s00 - w_k1_FDX_s002
                                 + 0.99 b_k1_FDX_s002 - 1.01 s_k1_FDX_s002  = 0
 w_k1_FDX_s003_return_constraint: 1.0845 w_k1_FDX_s00 - w_k1_FDX_s003
                                 + 0.99 b_k1_FDX_s003 - 1.01 s_k1_FDX_s003  = 0
 w_k1_FDX_s004_return_constraint: 0.890131 w_k1_FDX_s00 - w_k1_FDX_s004
                                 + 0.99 b_k1_FDX_s004 - 1.01 s_k1_FDX_s004  = 0
 w_k1_FDX_s005_return_constraint: 1.20503 w_k1_FDX_s00 - w_k1_FDX_s005
                                 + 0.99 b_k1_FDX_s005 - 1.01 s_k1_FDX_s005  = 0
 w_k1_FDX_s006_return_constraint: 1.12472 w_k1_FDX_s00 - w_k1_FDX_s006
                                 + 0.99 b_k1_FDX_s006 - 1.01 s_k1_FDX_s006  = 0
 w_k1_FDX_s007_return_constraint: 0.814758 w_k1_FDX_s00 - w_k1_FDX_s007
                                 + 0.99 b_k1_FDX_s007 - 1.01 s_k1_FDX_s007  = 0
 w_k1_FDX_s010_return_constraint: 1.32879 w_k1_FDX_s01 - w_k1_FDX_s010
                                 + 0.99 b_k1_FDX_s010 - 1.01 s_k1_FDX_s010  = 0
 w_k1_FDX_s011_return_constraint: 0.987241 w_k1_FDX_s01 - w_k1_FDX_s011
                                 + 0.99 b_k1_FDX_s011 - 1.01 s_k1_FDX_s011  = 0
 w_k1_FDX_s012_return_constraint: 0.646599 w_k1_FDX_s01 - w_k1_FDX_s012
                                 + 0.99 b_k1_FDX_s012 - 1.01 s_k1_FDX_s012  = 0
 w_k1_FDX_s013_return_constraint: 1.1493 w_k1_FDX_s01 - w_k1_FDX_s013
                                 + 0.99 b_k1_FDX_s013 - 1.01 s_k1_FDX_s013  = 0
 w_k1_FDX_s014_return_constraint: 1.24141 w_k1_FDX_s01 - w_k1_FDX_s014
                                 + 0.99 b_k1_FDX_s014 - 1.01 s_k1_FDX_s014  = 0
 w_k1_FDX_s015_return_constraint: 0.823515 w_k1_FDX_s01 - w_k1_FDX_s015
                                 + 0.99 b_k1_FDX_s015 - 1.01 s_k1_FDX_s015  = 0
 w_k1_FDX_s016_return_constraint: 1.55789 w_k1_FDX_s01 - w_k1_FDX_s016
                                 + 0.99 b_k1_FDX_s016 - 1.01 s_k1_FDX_s016  = 0
 w_k1_FDX_s017_return_constraint: 1.02354 w_k1_FDX_s01 - w_k1_FDX_s017
                                 + 0.99 b_k1_FDX_s017 - 1.01 s_k1_FDX_s017  = 0
 w_k1_FDX_s100_return_constraint: 0.841238 w_k1_FDX_s10 - w_k1_FDX_s100
                                 + 0.99 b_k1_FDX_s100 - 1.01 s_k1_FDX_s100  = 0
 w_k1_FDX_s101_return_constraint: 0.858335 w_k1_FDX_s10 - w_k1_FDX_s101
                                 + 0.99 b_k1_FDX_s101 - 1.01 s_k1_FDX_s101  = 0
 w_k1_FDX_s102_return_constraint: 0.914033 w_k1_FDX_s10 - w_k1_FDX_s102
                                 + 0.99 b_k1_FDX_s102 - 1.01 s_k1_FDX_s102  = 0
 w_k1_FDX_s103_return_constraint: 1.35946 w_k1_FDX_s10 - w_k1_FDX_s103
                                 + 0.99 b_k1_FDX_s103 - 1.01 s_k1_FDX_s103  = 0
 w_k1_FDX_s104_return_constraint: 0.819446 w_k1_FDX_s10 - w_k1_FDX_s104
                                 + 0.99 b_k1_FDX_s104 - 1.01 s_k1_FDX_s104  = 0
 w_k1_FDX_s105_return_constraint: 1.16856 w_k1_FDX_s10 - w_k1_FDX_s105
                                 + 0.99 b_k1_FDX_s105 - 1.01 s_k1_FDX_s105  = 0
 w_k1_FDX_s106_return_constraint: 1.21855 w_k1_FDX_s10 - w_k1_FDX_s106
                                 + 0.99 b_k1_FDX_s106 - 1.01 s_k1_FDX_s106  = 0
 w_k1_FDX_s107_return_constraint: 0.846191 w_k1_FDX_s10 - w_k1_FDX_s107
                                 + 0.99 b_k1_FDX_s107 - 1.01 s_k1_FDX_s107  = 0
 w_k1_FDX_s110_return_constraint: 1.41824 w_k1_FDX_s11 - w_k1_FDX_s110
                                 + 0.99 b_k1_FDX_s110 - 1.01 s_k1_FDX_s110  = 0
 w_k1_FDX_s111_return_constraint: 0.783625 w_k1_FDX_s11 - w_k1_FDX_s111
                                 + 0.99 b_k1_FDX_s111 - 1.01 s_k1_FDX_s111  = 0
 w_k1_FDX_s112_return_constraint: 0.862703 w_k1_FDX_s11 - w_k1_FDX_s112
                                 + 0.99 b_k1_FDX_s112 - 1.01 s_k1_FDX_s112  = 0
 w_k1_FDX_s113_return_constraint: 0.958802 w_k1_FDX_s11 - w_k1_FDX_s113
                                 + 0.99 b_k1_FDX_s113 - 1.01 s_k1_FDX_s113  = 0
 w_k1_FDX_s114_return_constraint: 0.997915 w_k1_FDX_s11 - w_k1_FDX_s114
                                 + 0.99 b_k1_FDX_s114 - 1.01 s_k1_FDX_s114  = 0
 w_k1_FDX_s115_return_constraint: 0.99454 w_k1_FDX_s11 - w_k1_FDX_s115
                                 + 0.99 b_k1_FDX_s115 - 1.01 s_k1_FDX_s115  = 0
 w_k1_FDX_s116_return_constraint: 1.23631 w_k1_FDX_s11 - w_k1_FDX_s116
                                 + 0.99 b_k1_FDX_s116 - 1.01 s_k1_FDX_s116  = 0
 w_k1_FDX_s117_return_constraint: 0.749865 w_k1_FDX_s11 - w_k1_FDX_s117
                                 + 0.99 b_k1_FDX_s117 - 1.01 s_k1_FDX_s117  = 0
 w_k1_IBM_s0_return_constraint:  0.919076 w_0_IBM - w_k1_IBM_s0
                                 + 0.99 b_k1_IBM_s0 - 1.01 s_k1_IBM_s0  = 0
 w_k1_IBM_s1_return_constraint:  1.27668 w_0_IBM - w_k1_IBM_s1
                                 + 0.99 b_k1_IBM_s1 - 1.01 s_k1_IBM_s1  = 0
 w_k1_IBM_s00_return_constraint: 1.37239 w_k1_IBM_s0 - w_k1_IBM_s00
                                 + 0.99 b_k1_IBM_s00 - 1.01 s_k1_IBM_s00  = 0
 w_k1_IBM_s01_return_constraint: 0.873979 w_k1_IBM_s0 - w_k1_IBM_s01
                                 + 0.99 b_k1_IBM_s01 - 1.01 s_k1_IBM_s01  = 0
 w_k1_IBM_s10_return_constraint: 1.23199 w_k1_IBM_s1 - w_k1_IBM_s10
                                 + 0.99 b_k1_IBM_s10 - 1.01 s_k1_IBM_s10  = 0
 w_k1_IBM_s11_return_constraint: 0.872302 w_k1_IBM_s1 - w_k1_IBM_s11
                                 + 0.99 b_k1_IBM_s11 - 1.01 s_k1_IBM_s11  = 0
 w_k1_IBM_s000_return_constraint: 0.867973 w_k1_IBM_s00 - w_k1_IBM_s000
                                 + 0.99 b_k1_IBM_s000 - 1.01 s_k1_IBM_s000  = 0
 w_k1_IBM_s001_return_constraint: 1.22877 w_k1_IBM_s00 - w_k1_IBM_s001
                                 + 0.99 b_k1_IBM_s001 - 1.01 s_k1_IBM_s001  = 0
 w_k1_IBM_s002_return_constraint: 1.36992 w_k1_IBM_s00 - w_k1_IBM_s002
                                 + 0.99 b_k1_IBM_s002 - 1.01 s_k1_IBM_s002  = 0
 w_k1_IBM_s003_return_constraint: 1.04387 w_k1_IBM_s00 - w_k1_IBM_s003
                                 + 0.99 b_k1_IBM_s003 - 1.01 s_k1_IBM_s003  = 0
 w_k1_IBM_s004_return_constraint: 1.15271 w_k1_IBM_s00 - w_k1_IBM_s004
                                 + 0.99 b_k1_IBM_s004 - 1.01 s_k1_IBM_s004  = 0
 w_k1_IBM_s005_return_constraint: 0.754474 w_k1_IBM_s00 - w_k1_IBM_s005
                                 + 0.99 b_k1_IBM_s005 - 1.01 s_k1_IBM_s005  = 0
 w_k1_IBM_s006_return_constraint: 1.66485 w_k1_IBM_s00 - w_k1_IBM_s006
                                 + 0.99 b_k1_IBM_s006 - 1.01 s_k1_IBM_s006  = 0
 w_k1_IBM_s007_return_constraint: 0.734386 w_k1_IBM_s00 - w_k1_IBM_s007
                                 + 0.99 b_k1_IBM_s007 - 1.01 s_k1_IBM_s007  = 0
 w_k1_IBM_s010_return_constraint: 1.72422 w_k1_IBM_s01 - w_k1_IBM_s010
                                 + 0.99 b_k1_IBM_s010 - 1.01 s_k1_IBM_s010  = 0
 w_k1_IBM_s011_return_constraint: 1.38562 w_k1_IBM_s01 - w_k1_IBM_s011
                                 + 0.99 b_k1_IBM_s011 - 1.01 s_k1_IBM_s011  = 0
 w_k1_IBM_s012_return_constraint: 0.717873 w_k1_IBM_s01 - w_k1_IBM_s012
                                 + 0.99 b_k1_IBM_s012 - 1.01 s_k1_IBM_s012  = 0
 w_k1_IBM_s013_return_constraint: 0.811925 w_k1_IBM_s01 - w_k1_IBM_s013
                                 + 0.99 b_k1_IBM_s013 - 1.01 s_k1_IBM_s013  = 0
 w_k1_IBM_s014_return_constraint: 1.1065 w_k1_IBM_s01 - w_k1_IBM_s014
                                 + 0.99 b_k1_IBM_s014 - 1.01 s_k1_IBM_s014  = 0
 w_k1_IBM_s015_return_constraint: 1.04688 w_k1_IBM_s01 - w_k1_IBM_s015
                                 + 0.99 b_k1_IBM_s015 - 1.01 s_k1_IBM_s015  = 0
 w_k1_IBM_s016_return_constraint: 0.914825 w_k1_IBM_s01 - w_k1_IBM_s016
                                 + 0.99 b_k1_IBM_s016 - 1.01 s_k1_IBM_s016  = 0
 w_k1_IBM_s017_return_constraint: 0.793341 w_k1_IBM_s01 - w_k1_IBM_s017
                                 + 0.99 b_k1_IBM_s017 - 1.01 s_k1_IBM_s017  = 0
 w_k1_IBM_s100_return_constraint: 0.821774 w_k1_IBM_s10 - w_k1_IBM_s100
                                 + 0.99 b_k1_IBM_s100 - 1.01 s_k1_IBM_s100  = 0
 w_k1_IBM_s101_return_constraint: 0.582416 w_k1_IBM_s10 - w_k1_IBM_s101
                                 + 0.99 b_k1_IBM_s101 - 1.01 s_k1_IBM_s101  = 0
 w_k1_IBM_s102_return_constraint: 1.30889 w_k1_IBM_s10 - w_k1_IBM_s102
                                 + 0.99 b_k1_IBM_s102 - 1.01 s_k1_IBM_s102  = 0
 w_k1_IBM_s103_return_constraint: 1.13703 w_k1_IBM_s10 - w_k1_IBM_s103
                                 + 0.99 b_k1_IBM_s103 - 1.01 s_k1_IBM_s103  = 0
 w_k1_IBM_s104_return_constraint: 0.803604 w_k1_IBM_s10 - w_k1_IBM_s104
                                 + 0.99 b_k1_IBM_s104 - 1.01 s_k1_IBM_s104  = 0
 w_k1_IBM_s105_return_constraint: 1.95271 w_k1_IBM_s10 - w_k1_IBM_s105
                                 + 0.99 b_k1_IBM_s105 - 1.01 s_k1_IBM_s105  = 0
 w_k1_IBM_s106_return_constraint: 0.683824 w_k1_IBM_s10 - w_k1_IBM_s106
                                 + 0.99 b_k1_IBM_s106 - 1.01 s_k1_IBM_s106  = 0
 w_k1_IBM_s107_return_constraint: 0.981706 w_k1_IBM_s10 - w_k1_IBM_s107
                                 + 0.99 b_k1_IBM_s107 - 1.01 s_k1_IBM_s107  = 0
 w_k1_IBM_s110_return_constraint: 1.46458 w_k1_IBM_s11 - w_k1_IBM_s110
                                 + 0.99 b_k1_IBM_s110 - 1.01 s_k1_IBM_s110  = 0
 w_k1_IBM_s111_return_constraint: 1.35267 w_k1_IBM_s11 - w_k1_IBM_s111
                                 + 0.99 b_k1_IBM_s111 - 1.01 s_k1_IBM_s111  = 0
 w_k1_IBM_s112_return_constraint: 0.982992 w_k1_IBM_s11 - w_k1_IBM_s112
                                 + 0.99 b_k1_IBM_s112 - 1.01 s_k1_IBM_s112  = 0
 w_k1_IBM_s113_return_constraint: 0.517579 w_k1_IBM_s11 - w_k1_IBM_s113
                                 + 0.99 b_k1_IBM_s113 - 1.01 s_k1_IBM_s113  = 0
 w_k1_IBM_s114_return_constraint: 1.68169 w_k1_IBM_s11 - w_k1_IBM_s114
                                 + 0.99 b_k1_IBM_s114 - 1.01 s_k1_IBM_s114  = 0
 w_k1_IBM_s115_return_constraint: 0.880403 w_k1_IBM_s11 - w_k1_IBM_s115
                                 + 0.99 b_k1_IBM_s115 - 1.01 s_k1_IBM_s115  = 0
 w_k1_IBM_s116_return_constraint: 1.06984 w_k1_IBM_s11 - w_k1_IBM_s116
                                 + 0.99 b_k1_IBM_s116 - 1.01 s_k1_IBM_s116  = 0
 w_k1_IBM_s117_return_constraint: 0.770968 w_k1_IBM_s11 - w_k1_IBM_s117
                                 + 0.99 b_k1_IBM_s117 - 1.01 s_k1_IBM_s117  = 0
 w_k1_INTC_s0_return_constraint: 0.976904 w_0_INTC - w_k1_INTC_s0
                                 + 0.99 b_k1_INTC_s0 - 1.01 s_k1_INTC_s0  = 0
 w_k1_INTC_s1_return_constraint: 1.07035 w_0_INTC - w_k1_INTC_s1
                                 + 0.99 b_k1_INTC_s1 - 1.01 s_k1_INTC_s1  = 0
 w_k1_INTC_s00_return_constraint: 1.15486 w_k1_INTC_s0 - w_k1_INTC_s00
                                 + 0.99 b_k1_INTC_s00 - 1.01 s_k1_INTC_s00  = 0
 w_k1_INTC_s01_return_constraint: 0.991644 w_k1_INTC_s0 - w_k1_INTC_s01
                                 + 0.99 b_k1_INTC_s01 - 1.01 s_k1_INTC_s01  = 0
 w_k1_INTC_s10_return_constraint: 1.0954 w_k1_INTC_s1 - w_k1_INTC_s10
                                 + 0.99 b_k1_INTC_s10 - 1.01 s_k1_INTC_s10  = 0
 w_k1_INTC_s11_return_constraint: 0.794114 w_k1_INTC_s1 - w_k1_INTC_s11
                                 + 0.99 b_k1_INTC_s11 - 1.01 s_k1_INTC_s11  = 0
 w_k1_INTC_s000_return_constraint: 0.953712 w_k1_INTC_s00 - w_k1_INTC_s000
                                 + 0.99 b_k1_INTC_s000 - 1.01 s_k1_INTC_s000
                                  = 0
 w_k1_INTC_s001_return_constraint: 1.19584 w_k1_INTC_s00 - w_k1_INTC_s001
                                 + 0.99 b_k1_INTC_s001 - 1.01 s_k1_INTC_s001
                                  = 0
 w_k1_INTC_s002_return_constraint: 0.985013 w_k1_INTC_s00 - w_k1_INTC_s002
                                 + 0.99 b_k1_INTC_s002 - 1.01 s_k1_INTC_s002
                                  = 0
 w_k1_INTC_s003_return_constraint: 1.13274 w_k1_INTC_s00 - w_k1_INTC_s003
                                 + 0.99 b_k1_INTC_s003 - 1.01 s_k1_INTC_s003
                                  = 0
 w_k1_INTC_s004_return_constraint: 1.03168 w_k1_INTC_s00 - w_k1_INTC_s004
                                 + 0.99 b_k1_INTC_s004 - 1.01 s_k1_INTC_s004
                                  = 0
 w_k1_INTC_s005_return_constraint: 1.21454 w_k1_INTC_s00 - w_k1_INTC_s005
                                 + 0.99 b_k1_INTC_s005 - 1.01 s_k1_INTC_s005
                                  = 0
 w_k1_INTC_s006_return_constraint: 1.29699 w_k1_INTC_s00 - w_k1_INTC_s006
                                 + 0.99 b_k1_INTC_s006 - 1.01 s_k1_INTC_s006
                                  = 0
 w_k1_INTC_s007_return_constraint: 1.01069 w_k1_INTC_s00 - w_k1_INTC_s007
                                 + 0.99 b_k1_INTC_s007 - 1.01 s_k1_INTC_s007
                                  = 0
 w_k1_INTC_s010_return_constraint: 1.11448 w_k1_INTC_s01 - w_k1_INTC_s010
                                 + 0.99 b_k1_INTC_s010 - 1.01 s_k1_INTC_s010
                                  = 0
 w_k1_INTC_s011_return_constraint: 0.991235 w_k1_INTC_s01 - w_k1_INTC_s011
                                 + 0.99 b_k1_INTC_s011 - 1.01 s_k1_INTC_s011
                                  = 0
 w_k1_INTC_s012_return_constraint: 0.700385 w_k1_INTC_s01 - w_k1_INTC_s012
                                 + 0.99 b_k1_INTC_s012 - 1.01 s_k1_INTC_s012
                                  = 0
 w_k1_INTC_s013_return_constraint: 0.98984 w_k1_INTC_s01 - w_k1_INTC_s013
                                 + 0.99 b_k1_INTC_s013 - 1.01 s_k1_INTC_s013
                                  = 0
 w_k1_INTC_s014_return_constraint: 1.19162 w_k1_INTC_s01 - w_k1_INTC_s014
                                 + 0.99 b_k1_INTC_s014 - 1.01 s_k1_INTC_s014
                                  = 0
 w_k1_INTC_s015_return_constraint: 0.799599 w_k1_INTC_s01 - w_k1_INTC_s015
                                 + 0.99 b_k1_INTC_s015 - 1.01 s_k1_INTC_s015
                                  = 0
 w_k1_INTC_s016_return_constraint: 1.34065 w_k1_INTC_s01 - w_k1_INTC_s016
                                 + 0.99 b_k1_INTC_s016 - 1.01 s_k1_INTC_s016
                                  = 0
 w_k1_INTC_s017_return_constraint: 1.00763 w_k1_INTC_s01 - w_k1_INTC_s017
                                 + 0.99 b_k1_INTC_s017 - 1.01 s_k1_INTC_s017
                                  = 0
 w_k1_INTC_s100_return_constraint: 0.89403 w_k1_INTC_s10 - w_k1_INTC_s100
                                 + 0.99 b_k1_INTC_s100 - 1.01 s_k1_INTC_s100
                                  = 0
 w_k1_INTC_s101_return_constraint: 0.886872 w_k1_INTC_s10 - w_k1_INTC_s101
                                 + 0.99 b_k1_INTC_s101 - 1.01 s_k1_INTC_s101
                                  = 0
 w_k1_INTC_s102_return_constraint: 0.949747 w_k1_INTC_s10 - w_k1_INTC_s102
                                 + 0.99 b_k1_INTC_s102 - 1.01 s_k1_INTC_s102
                                  = 0
 w_k1_INTC_s103_return_constraint: 1.1373 w_k1_INTC_s10 - w_k1_INTC_s103
                                 + 0.99 b_k1_INTC_s103 - 1.01 s_k1_INTC_s103
                                  = 0
 w_k1_INTC_s104_return_constraint: 0.931283 w_k1_INTC_s10 - w_k1_INTC_s104
                                 + 0.99 b_k1_INTC_s104 - 1.01 s_k1_INTC_s104
                                  = 0
 w_k1_INTC_s105_return_constraint: 1.13008 w_k1_INTC_s10 - w_k1_INTC_s105
                                 + 0.99 b_k1_INTC_s105 - 1.01 s_k1_INTC_s105
                                  = 0
 w_k1_INTC_s106_return_constraint: 1.04121 w_k1_INTC_s10 - w_k1_INTC_s106
                                 + 0.99 b_k1_INTC_s106 - 1.01 s_k1_INTC_s106
                                  = 0
 w_k1_INTC_s107_return_constraint: 0.957908 w_k1_INTC_s10 - w_k1_INTC_s107
                                 + 0.99 b_k1_INTC_s107 - 1.01 s_k1_INTC_s107
                                  = 0
 w_k1_INTC_s110_return_constraint: 1.45014 w_k1_INTC_s11 - w_k1_INTC_s110
                                 + 0.99 b_k1_INTC_s110 - 1.01 s_k1_INTC_s110
                                  = 0
 w_k1_INTC_s111_return_constraint: 0.85223 w_k1_INTC_s11 - w_k1_INTC_s111
                                 + 0.99 b_k1_INTC_s111 - 1.01 s_k1_INTC_s111
                                  = 0
 w_k1_INTC_s112_return_constraint: 0.972422 w_k1_INTC_s11 - w_k1_INTC_s112
                                 + 0.99 b_k1_INTC_s112 - 1.01 s_k1_INTC_s112
                                  = 0
 w_k1_INTC_s113_return_constraint: 0.877422 w_k1_INTC_s11 - w_k1_INTC_s113
                                 + 0.99 b_k1_INTC_s113 - 1.01 s_k1_INTC_s113
                                  = 0
 w_k1_INTC_s114_return_constraint: 1.04395 w_k1_INTC_s11 - w_k1_INTC_s114
                                 + 0.99 b_k1_INTC_s114 - 1.01 s_k1_INTC_s114
                                  = 0
 w_k1_INTC_s115_return_constraint: 1.06597 w_k1_INTC_s11 - w_k1_INTC_s115
                                 + 0.99 b_k1_INTC_s115 - 1.01 s_k1_INTC_s115
                                  = 0
 w_k1_INTC_s116_return_constraint: 1.13371 w_k1_INTC_s11 - w_k1_INTC_s116
                                 + 0.99 b_k1_INTC_s116 - 1.01 s_k1_INTC_s116
                                  = 0
 w_k1_INTC_s117_return_constraint: 0.806734 w_k1_INTC_s11 - w_k1_INTC_s117
                                 + 0.99 b_k1_INTC_s117 - 1.01 s_k1_INTC_s117
                                  = 0
 w_k1_KO_s0_return_constraint:   0.93528 w_0_KO - w_k1_KO_s0 + 0.99 b_k1_KO_s0
                                 - 1.01 s_k1_KO_s0  = 0
 w_k1_KO_s1_return_constraint:   1.11678 w_0_KO - w_k1_KO_s1 + 0.99 b_k1_KO_s1
                                 - 1.01 s_k1_KO_s1  = 0
 w_k1_KO_s00_return_constraint:  1.44363 w_k1_KO_s0 - w_k1_KO_s00
                                 + 0.99 b_k1_KO_s00 - 1.01 s_k1_KO_s00  = 0
 w_k1_KO_s01_return_constraint:  0.973752 w_k1_KO_s0 - w_k1_KO_s01
                                 + 0.99 b_k1_KO_s01 - 1.01 s_k1_KO_s01  = 0
 w_k1_KO_s10_return_constraint:  1.26189 w_k1_KO_s1 - w_k1_KO_s10
                                 + 0.99 b_k1_KO_s10 - 1.01 s_k1_KO_s10  = 0
 w_k1_KO_s11_return_constraint:  0.795551 w_k1_KO_s1 - w_k1_KO_s11
                                 + 0.99 b_k1_KO_s11 - 1.01 s_k1_KO_s11  = 0
 w_k1_KO_s000_return_constraint: 0.830447 w_k1_KO_s00 - w_k1_KO_s000
                                 + 0.99 b_k1_KO_s000 - 1.01 s_k1_KO_s000  = 0
 w_k1_KO_s001_return_constraint: 1.43528 w_k1_KO_s00 - w_k1_KO_s001
                                 + 0.99 b_k1_KO_s001 - 1.01 s_k1_KO_s001  = 0
 w_k1_KO_s002_return_constraint: 0.937812 w_k1_KO_s00 - w_k1_KO_s002
                                 + 0.99 b_k1_KO_s002 - 1.01 s_k1_KO_s002  = 0
 w_k1_KO_s003_return_constraint: 0.968376 w_k1_KO_s00 - w_k1_KO_s003
                                 + 0.99 b_k1_KO_s003 - 1.01 s_k1_KO_s003  = 0
 w_k1_KO_s004_return_constraint: 0.933405 w_k1_KO_s00 - w_k1_KO_s004
                                 + 0.99 b_k1_KO_s004 - 1.01 s_k1_KO_s004  = 0
 w_k1_KO_s005_return_constraint: 0.92962 w_k1_KO_s00 - w_k1_KO_s005
                                 + 0.99 b_k1_KO_s005 - 1.01 s_k1_KO_s005  = 0
 w_k1_KO_s006_return_constraint: 1.44948 w_k1_KO_s00 - w_k1_KO_s006
                                 + 0.99 b_k1_KO_s006 - 1.01 s_k1_KO_s006  = 0
 w_k1_KO_s007_return_constraint: 0.700761 w_k1_KO_s00 - w_k1_KO_s007
                                 + 0.99 b_k1_KO_s007 - 1.01 s_k1_KO_s007  = 0
 w_k1_KO_s010_return_constraint: 1.35642 w_k1_KO_s01 - w_k1_KO_s010
                                 + 0.99 b_k1_KO_s010 - 1.01 s_k1_KO_s010  = 0
 w_k1_KO_s011_return_constraint: 1.18969 w_k1_KO_s01 - w_k1_KO_s011
                                 + 0.99 b_k1_KO_s011 - 1.01 s_k1_KO_s011  = 0
 w_k1_KO_s012_return_constraint: 0.616384 w_k1_KO_s01 - w_k1_KO_s012
                                 + 0.99 b_k1_KO_s012 - 1.01 s_k1_KO_s012  = 0
 w_k1_KO_s013_return_constraint: 0.772383 w_k1_KO_s01 - w_k1_KO_s013
                                 + 0.99 b_k1_KO_s013 - 1.01 s_k1_KO_s013  = 0
 w_k1_KO_s014_return_constraint: 1.12305 w_k1_KO_s01 - w_k1_KO_s014
                                 + 0.99 b_k1_KO_s014 - 1.01 s_k1_KO_s014  = 0
 w_k1_KO_s015_return_constraint: 0.89939 w_k1_KO_s01 - w_k1_KO_s015
                                 + 0.99 b_k1_KO_s015 - 1.01 s_k1_KO_s015  = 0
 w_k1_KO_s016_return_constraint: 1.18624 w_k1_KO_s01 - w_k1_KO_s016
                                 + 0.99 b_k1_KO_s016 - 1.01 s_k1_KO_s016  = 0
 w_k1_KO_s017_return_constraint: 0.852576 w_k1_KO_s01 - w_k1_KO_s017
                                 + 0.99 b_k1_KO_s017 - 1.01 s_k1_KO_s017  = 0
 w_k1_KO_s100_return_constraint: 0.863194 w_k1_KO_s10 - w_k1_KO_s100
                                 + 0.99 b_k1_KO_s100 - 1.01 s_k1_KO_s100  = 0
 w_k1_KO_s101_return_constraint: 0.653944 w_k1_KO_s10 - w_k1_KO_s101
                                 + 0.99 b_k1_KO_s101 - 1.01 s_k1_KO_s101  = 0
 w_k1_KO_s102_return_constraint: 1.09602 w_k1_KO_s10 - w_k1_KO_s102
                                 + 0.99 b_k1_KO_s102 - 1.01 s_k1_KO_s102  = 0
 w_k1_KO_s103_return_constraint: 1.20504 w_k1_KO_s10 - w_k1_KO_s103
                                 + 0.99 b_k1_KO_s103 - 1.01 s_k1_KO_s103  = 0
 w_k1_KO_s104_return_constraint: 0.862174 w_k1_KO_s10 - w_k1_KO_s104
                                 + 0.99 b_k1_KO_s104 - 1.01 s_k1_KO_s104  = 0
 w_k1_KO_s105_return_constraint: 1.47695 w_k1_KO_s10 - w_k1_KO_s105
                                 + 0.99 b_k1_KO_s105 - 1.01 s_k1_KO_s105  = 0
 w_k1_KO_s106_return_constraint: 0.918623 w_k1_KO_s10 - w_k1_KO_s106
                                 + 0.99 b_k1_KO_s106 - 1.01 s_k1_KO_s106  = 0
 w_k1_KO_s107_return_constraint: 0.948052 w_k1_KO_s10 - w_k1_KO_s107
                                 + 0.99 b_k1_KO_s107 - 1.01 s_k1_KO_s107  = 0
 w_k1_KO_s110_return_constraint: 1.6078 w_k1_KO_s11 - w_k1_KO_s110
                                 + 0.99 b_k1_KO_s110 - 1.01 s_k1_KO_s110  = 0
 w_k1_KO_s111_return_constraint: 1.19557 w_k1_KO_s11 - w_k1_KO_s111
                                 + 0.99 b_k1_KO_s111 - 1.01 s_k1_KO_s111  = 0
 w_k1_KO_s112_return_constraint: 0.976949 w_k1_KO_s11 - w_k1_KO_s112
                                 + 0.99 b_k1_KO_s112 - 1.01 s_k1_KO_s112  = 0
 w_k1_KO_s113_return_constraint: 0.794281 w_k1_KO_s11 - w_k1_KO_s113
                                 + 0.99 b_k1_KO_s113 - 1.01 s_k1_KO_s113  = 0
 w_k1_KO_s114_return_constraint: 1.32468 w_k1_KO_s11 - w_k1_KO_s114
                                 + 0.99 b_k1_KO_s114 - 1.01 s_k1_KO_s114  = 0
 w_k1_KO_s115_return_constraint: 1.09298 w_k1_KO_s11 - w_k1_KO_s115
                                 + 0.99 b_k1_KO_s115 - 1.01 s_k1_KO_s115  = 0
 w_k1_KO_s116_return_constraint: 1.16913 w_k1_KO_s11 - w_k1_KO_s116
                                 + 0.99 b_k1_KO_s116 - 1.01 s_k1_KO_s116  = 0
 w_k1_KO_s117_return_constraint: 0.814679 w_k1_KO_s11 - w_k1_KO_s117
                                 + 0.99 b_k1_KO_s117 - 1.01 s_k1_KO_s117  = 0
 w_k1_MAT_s0_return_constraint:  0.897471 w_0_MAT - w_k1_MAT_s0
                                 + 0.99 b_k1_MAT_s0 - 1.01 s_k1_MAT_s0  = 0
 w_k1_MAT_s1_return_constraint:  1.21927 w_0_MAT - w_k1_MAT_s1
                                 + 0.99 b_k1_MAT_s1 - 1.01 s_k1_MAT_s1  = 0
 w_k1_MAT_s00_return_constraint: 1.66861 w_k1_MAT_s0 - w_k1_MAT_s00
                                 + 0.99 b_k1_MAT_s00 - 1.01 s_k1_MAT_s00  = 0
 w_k1_MAT_s01_return_constraint: 0.91944 w_k1_MAT_s0 - w_k1_MAT_s01
                                 + 0.99 b_k1_MAT_s01 - 1.01 s_k1_MAT_s01  = 0
 w_k1_MAT_s10_return_constraint: 1.15585 w_k1_MAT_s1 - w_k1_MAT_s10
                                 + 0.99 b_k1_MAT_s10 - 1.01 s_k1_MAT_s10  = 0
 w_k1_MAT_s11_return_constraint: 0.880388 w_k1_MAT_s1 - w_k1_MAT_s11
                                 + 0.99 b_k1_MAT_s11 - 1.01 s_k1_MAT_s11  = 0
 w_k1_MAT_s000_return_constraint: 0.730612 w_k1_MAT_s00 - w_k1_MAT_s000
                                 + 0.99 b_k1_MAT_s000 - 1.01 s_k1_MAT_s000  = 0
 w_k1_MAT_s001_return_constraint: 1.1826 w_k1_MAT_s00 - w_k1_MAT_s001
                                 + 0.99 b_k1_MAT_s001 - 1.01 s_k1_MAT_s001  = 0
 w_k1_MAT_s002_return_constraint: 1.10804 w_k1_MAT_s00 - w_k1_MAT_s002
                                 + 0.99 b_k1_MAT_s002 - 1.01 s_k1_MAT_s002  = 0
 w_k1_MAT_s003_return_constraint: 0.87296 w_k1_MAT_s00 - w_k1_MAT_s003
                                 + 0.99 b_k1_MAT_s003 - 1.01 s_k1_MAT_s003  = 0
 w_k1_MAT_s004_return_constraint: 0.947664 w_k1_MAT_s00 - w_k1_MAT_s004
                                 + 0.99 b_k1_MAT_s004 - 1.01 s_k1_MAT_s004  = 0
 w_k1_MAT_s005_return_constraint: 0.81151 w_k1_MAT_s00 - w_k1_MAT_s005
                                 + 0.99 b_k1_MAT_s005 - 1.01 s_k1_MAT_s005  = 0
 w_k1_MAT_s006_return_constraint: 1.54517 w_k1_MAT_s00 - w_k1_MAT_s006
                                 + 0.99 b_k1_MAT_s006 - 1.01 s_k1_MAT_s006  = 0
 w_k1_MAT_s007_return_constraint: 0.567597 w_k1_MAT_s00 - w_k1_MAT_s007
                                 + 0.99 b_k1_MAT_s007 - 1.01 s_k1_MAT_s007  = 0
 w_k1_MAT_s010_return_constraint: 1.74141 w_k1_MAT_s01 - w_k1_MAT_s010
                                 + 0.99 b_k1_MAT_s010 - 1.01 s_k1_MAT_s010  = 0
 w_k1_MAT_s011_return_constraint: 1.43951 w_k1_MAT_s01 - w_k1_MAT_s011
                                 + 0.99 b_k1_MAT_s011 - 1.01 s_k1_MAT_s011  = 0
 w_k1_MAT_s012_return_constraint: 0.654421 w_k1_MAT_s01 - w_k1_MAT_s012
                                 + 0.99 b_k1_MAT_s012 - 1.01 s_k1_MAT_s012  = 0
 w_k1_MAT_s013_return_constraint: 0.72608 w_k1_MAT_s01 - w_k1_MAT_s013
                                 + 0.99 b_k1_MAT_s013 - 1.01 s_k1_MAT_s013  = 0
 w_k1_MAT_s014_return_constraint: 1.12503 w_k1_MAT_s01 - w_k1_MAT_s014
                                 + 0.99 b_k1_MAT_s014 - 1.01 s_k1_MAT_s014  = 0
 w_k1_MAT_s015_return_constraint: 0.897889 w_k1_MAT_s01 - w_k1_MAT_s015
                                 + 0.99 b_k1_MAT_s015 - 1.01 s_k1_MAT_s015  = 0
 w_k1_MAT_s016_return_constraint: 0.951579 w_k1_MAT_s01 - w_k1_MAT_s016
                                 + 0.99 b_k1_MAT_s016 - 1.01 s_k1_MAT_s016  = 0
 w_k1_MAT_s017_return_constraint: 0.803732 w_k1_MAT_s01 - w_k1_MAT_s017
                                 + 0.99 b_k1_MAT_s017 - 1.01 s_k1_MAT_s017  = 0
 w_k1_MAT_s100_return_constraint: 0.63611 w_k1_MAT_s10 - w_k1_MAT_s100
                                 + 0.99 b_k1_MAT_s100 - 1.01 s_k1_MAT_s100  = 0
 w_k1_MAT_s101_return_constraint: 0.594609 w_k1_MAT_s10 - w_k1_MAT_s101
                                 + 0.99 b_k1_MAT_s101 - 1.01 s_k1_MAT_s101  = 0
 w_k1_MAT_s102_return_constraint: 1.64198 w_k1_MAT_s10 - w_k1_MAT_s102
                                 + 0.99 b_k1_MAT_s102 - 1.01 s_k1_MAT_s102  = 0
 w_k1_MAT_s103_return_constraint: 1.63371 w_k1_MAT_s10 - w_k1_MAT_s103
                                 + 0.99 b_k1_MAT_s103 - 1.01 s_k1_MAT_s103  = 0
 w_k1_MAT_s104_return_constraint: 1.03097 w_k1_MAT_s10 - w_k1_MAT_s104
                                 + 0.99 b_k1_MAT_s104 - 1.01 s_k1_MAT_s104  = 0
 w_k1_MAT_s105_return_constraint: 2.1705 w_k1_MAT_s10 - w_k1_MAT_s105
                                 + 0.99 b_k1_MAT_s105 - 1.01 s_k1_MAT_s105  = 0
 w_k1_MAT_s106_return_constraint: 0.779624 w_k1_MAT_s10 - w_k1_MAT_s106
                                 + 0.99 b_k1_MAT_s106 - 1.01 s_k1_MAT_s106  = 0
 w_k1_MAT_s107_return_constraint: 1.15811 w_k1_MAT_s10 - w_k1_MAT_s107
                                 + 0.99 b_k1_MAT_s107 - 1.01 s_k1_MAT_s107  = 0
 w_k1_MAT_s110_return_constraint: 1.63614 w_k1_MAT_s11 - w_k1_MAT_s110
                                 + 0.99 b_k1_MAT_s110 - 1.01 s_k1_MAT_s110  = 0
 w_k1_MAT_s111_return_constraint: 1.21978 w_k1_MAT_s11 - w_k1_MAT_s111
                                 + 0.99 b_k1_MAT_s111 - 1.01 s_k1_MAT_s111  = 0
 w_k1_MAT_s112_return_constraint: 0.967129 w_k1_MAT_s11 - w_k1_MAT_s112
                                 + 0.99 b_k1_MAT_s112 - 1.01 s_k1_MAT_s112  = 0
 w_k1_MAT_s113_return_constraint: 0.599319 w_k1_MAT_s11 - w_k1_MAT_s113
                                 + 0.99 b_k1_MAT_s113 - 1.01 s_k1_MAT_s113  = 0
 w_k1_MAT_s114_return_constraint: 1.40814 w_k1_MAT_s11 - w_k1_MAT_s114
                                 + 0.99 b_k1_MAT_s114 - 1.01 s_k1_MAT_s114  = 0
 w_k1_MAT_s115_return_constraint: 1.06066 w_k1_MAT_s11 - w_k1_MAT_s115
                                 + 0.99 b_k1_MAT_s115 - 1.01 s_k1_MAT_s115  = 0
 w_k1_MAT_s116_return_constraint: 1.2503 w_k1_MAT_s11 - w_k1_MAT_s116
                                 + 0.99 b_k1_MAT_s116 - 1.01 s_k1_MAT_s116  = 0
 w_k1_MAT_s117_return_constraint: 0.701926 w_k1_MAT_s11 - w_k1_MAT_s117
                                 + 0.99 b_k1_MAT_s117 - 1.01 s_k1_MAT_s117  = 0
 w_k1_MSFT_s0_return_constraint: 1.00523 w_0_MSFT - w_k1_MSFT_s0
                                 + 0.99 b_k1_MSFT_s0 - 1.01 s_k1_MSFT_s0  = 0
 w_k1_MSFT_s1_return_constraint: 1.02254 w_0_MSFT - w_k1_MSFT_s1
                                 + 0.99 b_k1_MSFT_s1 - 1.01 s_k1_MSFT_s1  = 0
 w_k1_MSFT_s00_return_constraint: 1.23501 w_k1_MSFT_s0 - w_k1_MSFT_s00
                                 + 0.99 b_k1_MSFT_s00 - 1.01 s_k1_MSFT_s00  = 0
 w_k1_MSFT_s01_return_constraint: 0.999554 w_k1_MSFT_s0 - w_k1_MSFT_s01
                                 + 0.99 b_k1_MSFT_s01 - 1.01 s_k1_MSFT_s01  = 0
 w_k1_MSFT_s10_return_constraint: 1.15854 w_k1_MSFT_s1 - w_k1_MSFT_s10
                                 + 0.99 b_k1_MSFT_s10 - 1.01 s_k1_MSFT_s10  = 0
 w_k1_MSFT_s11_return_constraint: 0.874025 w_k1_MSFT_s1 - w_k1_MSFT_s11
                                 + 0.99 b_k1_MSFT_s11 - 1.01 s_k1_MSFT_s11  = 0
 w_k1_MSFT_s000_return_constraint: 0.899696 w_k1_MSFT_s00 - w_k1_MSFT_s000
                                 + 0.99 b_k1_MSFT_s000 - 1.01 s_k1_MSFT_s000
                                  = 0
 w_k1_MSFT_s001_return_constraint: 1.23543 w_k1_MSFT_s00 - w_k1_MSFT_s001
                                 + 0.99 b_k1_MSFT_s001 - 1.01 s_k1_MSFT_s001
                                  = 0
 w_k1_MSFT_s002_return_constraint: 0.798542 w_k1_MSFT_s00 - w_k1_MSFT_s002
                                 + 0.99 b_k1_MSFT_s002 - 1.01 s_k1_MSFT_s002
                                  = 0
 w_k1_MSFT_s003_return_constraint: 1.04635 w_k1_MSFT_s00 - w_k1_MSFT_s003
                                 + 0.99 b_k1_MSFT_s003 - 1.01 s_k1_MSFT_s003
                                  = 0
 w_k1_MSFT_s004_return_constraint: 0.89834 w_k1_MSFT_s00 - w_k1_MSFT_s004
                                 + 0.99 b_k1_MSFT_s004 - 1.01 s_k1_MSFT_s004
                                  = 0
 w_k1_MSFT_s005_return_constraint: 1.11032 w_k1_MSFT_s00 - w_k1_MSFT_s005
                                 + 0.99 b_k1_MSFT_s005 - 1.01 s_k1_MSFT_s005
                                  = 0
 w_k1_MSFT_s006_return_constraint: 1.16361 w_k1_MSFT_s00 - w_k1_MSFT_s006
                                 + 0.99 b_k1_MSFT_s006 - 1.01 s_k1_MSFT_s006
                                  = 0
 w_k1_MSFT_s007_return_constraint: 0.870693 w_k1_MSFT_s00 - w_k1_MSFT_s007
                                 + 0.99 b_k1_MSFT_s007 - 1.01 s_k1_MSFT_s007
                                  = 0
 w_k1_MSFT_s010_return_constraint: 1.18018 w_k1_MSFT_s01 - w_k1_MSFT_s010
                                 + 0.99 b_k1_MSFT_s010 - 1.01 s_k1_MSFT_s010
                                  = 0
 w_k1_MSFT_s011_return_constraint: 1.02015 w_k1_MSFT_s01 - w_k1_MSFT_s011
                                 + 0.99 b_k1_MSFT_s011 - 1.01 s_k1_MSFT_s011
                                  = 0
 w_k1_MSFT_s012_return_constraint: 0.688816 w_k1_MSFT_s01 - w_k1_MSFT_s012
                                 + 0.99 b_k1_MSFT_s012 - 1.01 s_k1_MSFT_s012
                                  = 0
 w_k1_MSFT_s013_return_constraint: 0.984553 w_k1_MSFT_s01 - w_k1_MSFT_s013
                                 + 0.99 b_k1_MSFT_s013 - 1.01 s_k1_MSFT_s013
                                  = 0
 w_k1_MSFT_s014_return_constraint: 1.16127 w_k1_MSFT_s01 - w_k1_MSFT_s014
                                 + 0.99 b_k1_MSFT_s014 - 1.01 s_k1_MSFT_s014
                                  = 0
 w_k1_MSFT_s015_return_constraint: 0.821318 w_k1_MSFT_s01 - w_k1_MSFT_s015
                                 + 0.99 b_k1_MSFT_s015 - 1.01 s_k1_MSFT_s015
                                  = 0
 w_k1_MSFT_s016_return_constraint: 1.4576 w_k1_MSFT_s01 - w_k1_MSFT_s016
                                 + 0.99 b_k1_MSFT_s016 - 1.01 s_k1_MSFT_s016
                                  = 0
 w_k1_MSFT_s017_return_constraint: 0.955863 w_k1_MSFT_s01 - w_k1_MSFT_s017
                                 + 0.99 b_k1_MSFT_s017 - 1.01 s_k1_MSFT_s017
                                  = 0
 w_k1_MSFT_s100_return_constraint: 0.896033 w_k1_MSFT_s10 - w_k1_MSFT_s100
                                 + 0.99 b_k1_MSFT_s100 - 1.01 s_k1_MSFT_s100
                                  = 0
 w_k1_MSFT_s101_return_constraint: 0.802258 w_k1_MSFT_s10 - w_k1_MSFT_s101
                                 + 0.99 b_k1_MSFT_s101 - 1.01 s_k1_MSFT_s101
                                  = 0
 w_k1_MSFT_s102_return_constraint: 1.01261 w_k1_MSFT_s10 - w_k1_MSFT_s102
                                 + 0.99 b_k1_MSFT_s102 - 1.01 s_k1_MSFT_s102
                                  = 0
 w_k1_MSFT_s103_return_constraint: 1.38292 w_k1_MSFT_s10 - w_k1_MSFT_s103
                                 + 0.99 b_k1_MSFT_s103 - 1.01 s_k1_MSFT_s103
                                  = 0
 w_k1_MSFT_s104_return_constraint: 0.9392 w_k1_MSFT_s10 - w_k1_MSFT_s104
                                 + 0.99 b_k1_MSFT_s104 - 1.01 s_k1_MSFT_s104
                                  = 0
 w_k1_MSFT_s105_return_constraint: 1.30296 w_k1_MSFT_s10 - w_k1_MSFT_s105
                                 + 0.99 b_k1_MSFT_s105 - 1.01 s_k1_MSFT_s105
                                  = 0
 w_k1_MSFT_s106_return_constraint: 1.21627 w_k1_MSFT_s10 - w_k1_MSFT_s106
                                 + 0.99 b_k1_MSFT_s106 - 1.01 s_k1_MSFT_s106
                                  = 0
 w_k1_MSFT_s107_return_constraint: 0.89955 w_k1_MSFT_s10 - w_k1_MSFT_s107
                                 + 0.99 b_k1_MSFT_s107 - 1.01 s_k1_MSFT_s107
                                  = 0
 w_k1_MSFT_s110_return_constraint: 1.24705 w_k1_MSFT_s11 - w_k1_MSFT_s110
                                 + 0.99 b_k1_MSFT_s110 - 1.01 s_k1_MSFT_s110
                                  = 0
 w_k1_MSFT_s111_return_constraint: 0.795181 w_k1_MSFT_s11 - w_k1_MSFT_s111
                                 + 0.99 b_k1_MSFT_s111 - 1.01 s_k1_MSFT_s111
                                  = 0
 w_k1_MSFT_s112_return_constraint: 0.866507 w_k1_MSFT_s11 - w_k1_MSFT_s112
                                 + 0.99 b_k1_MSFT_s112 - 1.01 s_k1_MSFT_s112
                                  = 0
 w_k1_MSFT_s113_return_constraint: 0.962766 w_k1_MSFT_s11 - w_k1_MSFT_s113
                                 + 0.99 b_k1_MSFT_s113 - 1.01 s_k1_MSFT_s113
                                  = 0
 w_k1_MSFT_s114_return_constraint: 0.866862 w_k1_MSFT_s11 - w_k1_MSFT_s114
                                 + 0.99 b_k1_MSFT_s114 - 1.01 s_k1_MSFT_s114
                                  = 0
 w_k1_MSFT_s115_return_constraint: 0.957577 w_k1_MSFT_s11 - w_k1_MSFT_s115
                                 + 0.99 b_k1_MSFT_s115 - 1.01 s_k1_MSFT_s115
                                  = 0
 w_k1_MSFT_s116_return_constraint: 1.06931 w_k1_MSFT_s11 - w_k1_MSFT_s116
                                 + 0.99 b_k1_MSFT_s116 - 1.01 s_k1_MSFT_s116
                                  = 0
 w_k1_MSFT_s117_return_constraint: 0.734103 w_k1_MSFT_s11 - w_k1_MSFT_s117
                                 + 0.99 b_k1_MSFT_s117 - 1.01 s_k1_MSFT_s117
                                  = 0
 w_k1_PG_s0_return_constraint:   0.975656 w_0_PG - w_k1_PG_s0 + 0.99 b_k1_PG_s0
                                 - 1.01 s_k1_PG_s0  = 0
 w_k1_PG_s1_return_constraint:   1.03086 w_0_PG - w_k1_PG_s1 + 0.99 b_k1_PG_s1
                                 - 1.01 s_k1_PG_s1  = 0
 w_k1_PG_s00_return_constraint:  1.25331 w_k1_PG_s0 - w_k1_PG_s00
                                 + 0.99 b_k1_PG_s00 - 1.01 s_k1_PG_s00  = 0
 w_k1_PG_s01_return_constraint:  0.942001 w_k1_PG_s0 - w_k1_PG_s01
                                 + 0.99 b_k1_PG_s01 - 1.01 s_k1_PG_s01  = 0
 w_k1_PG_s10_return_constraint:  1.1102 w_k1_PG_s1 - w_k1_PG_s10
                                 + 0.99 b_k1_PG_s10 - 1.01 s_k1_PG_s10  = 0
 w_k1_PG_s11_return_constraint:  0.920326 w_k1_PG_s1 - w_k1_PG_s11
                                 + 0.99 b_k1_PG_s11 - 1.01 s_k1_PG_s11  = 0
 w_k1_PG_s000_return_constraint: 0.882421 w_k1_PG_s00 - w_k1_PG_s000
                                 + 0.99 b_k1_PG_s000 - 1.01 s_k1_PG_s000  = 0
 w_k1_PG_s001_return_constraint: 1.17483 w_k1_PG_s00 - w_k1_PG_s001
                                 + 0.99 b_k1_PG_s001 - 1.01 s_k1_PG_s001  = 0
 w_k1_PG_s002_return_constraint: 0.920964 w_k1_PG_s00 - w_k1_PG_s002
                                 + 0.99 b_k1_PG_s002 - 1.01 s_k1_PG_s002  = 0
 w_k1_PG_s003_return_constraint: 0.998476 w_k1_PG_s00 - w_k1_PG_s003
                                 + 0.99 b_k1_PG_s003 - 1.01 s_k1_PG_s003  = 0
 w_k1_PG_s004_return_constraint: 0.89555 w_k1_PG_s00 - w_k1_PG_s004
                                 + 0.99 b_k1_PG_s004 - 1.01 s_k1_PG_s004  = 0
 w_k1_PG_s005_return_constraint: 1.00949 w_k1_PG_s00 - w_k1_PG_s005
                                 + 0.99 b_k1_PG_s005 - 1.01 s_k1_PG_s005  = 0
 w_k1_PG_s006_return_constraint: 1.10022 w_k1_PG_s00 - w_k1_PG_s006
                                 + 0.99 b_k1_PG_s006 - 1.01 s_k1_PG_s006  = 0
 w_k1_PG_s007_return_constraint: 0.810661 w_k1_PG_s00 - w_k1_PG_s007
                                 + 0.99 b_k1_PG_s007 - 1.01 s_k1_PG_s007  = 0
 w_k1_PG_s010_return_constraint: 1.12985 w_k1_PG_s01 - w_k1_PG_s010
                                 + 0.99 b_k1_PG_s010 - 1.01 s_k1_PG_s010  = 0
 w_k1_PG_s011_return_constraint: 1.17783 w_k1_PG_s01 - w_k1_PG_s011
                                 + 0.99 b_k1_PG_s011 - 1.01 s_k1_PG_s011  = 0
 w_k1_PG_s012_return_constraint: 0.704995 w_k1_PG_s01 - w_k1_PG_s012
                                 + 0.99 b_k1_PG_s012 - 1.01 s_k1_PG_s012  = 0
 w_k1_PG_s013_return_constraint: 0.978673 w_k1_PG_s01 - w_k1_PG_s013
                                 + 0.99 b_k1_PG_s013 - 1.01 s_k1_PG_s013  = 0
 w_k1_PG_s014_return_constraint: 1.15505 w_k1_PG_s01 - w_k1_PG_s014
                                 + 0.99 b_k1_PG_s014 - 1.01 s_k1_PG_s014  = 0
 w_k1_PG_s015_return_constraint: 0.902108 w_k1_PG_s01 - w_k1_PG_s015
                                 + 0.99 b_k1_PG_s015 - 1.01 s_k1_PG_s015  = 0
 w_k1_PG_s016_return_constraint: 1.31014 w_k1_PG_s01 - w_k1_PG_s016
                                 + 0.99 b_k1_PG_s016 - 1.01 s_k1_PG_s016  = 0
 w_k1_PG_s017_return_constraint: 0.988989 w_k1_PG_s01 - w_k1_PG_s017
                                 + 0.99 b_k1_PG_s017 - 1.01 s_k1_PG_s017  = 0
 w_k1_PG_s100_return_constraint: 0.86304 w_k1_PG_s10 - w_k1_PG_s100
                                 + 0.99 b_k1_PG_s100 - 1.01 s_k1_PG_s100  = 0
 w_k1_PG_s101_return_constraint: 0.899289 w_k1_PG_s10 - w_k1_PG_s101
                                 + 0.99 b_k1_PG_s101 - 1.01 s_k1_PG_s101  = 0
 w_k1_PG_s102_return_constraint: 1.11743 w_k1_PG_s10 - w_k1_PG_s102
                                 + 0.99 b_k1_PG_s102 - 1.01 s_k1_PG_s102  = 0
 w_k1_PG_s103_return_constraint: 1.28311 w_k1_PG_s10 - w_k1_PG_s103
                                 + 0.99 b_k1_PG_s103 - 1.01 s_k1_PG_s103  = 0
 w_k1_PG_s104_return_constraint: 0.954603 w_k1_PG_s10 - w_k1_PG_s104
                                 + 0.99 b_k1_PG_s104 - 1.01 s_k1_PG_s104  = 0
 w_k1_PG_s105_return_constraint: 1.38831 w_k1_PG_s10 - w_k1_PG_s105
                                 + 0.99 b_k1_PG_s105 - 1.01 s_k1_PG_s105  = 0
 w_k1_PG_s106_return_constraint: 1.11644 w_k1_PG_s10 - w_k1_PG_s106
                                 + 0.99 b_k1_PG_s106 - 1.01 s_k1_PG_s106  = 0
 w_k1_PG_s107_return_constraint: 0.943306 w_k1_PG_s10 - w_k1_PG_s107
                                 + 0.99 b_k1_PG_s107 - 1.01 s_k1_PG_s107  = 0
 w_k1_PG_s110_return_constraint: 1.35865 w_k1_PG_s11 - w_k1_PG_s110
                                 + 0.99 b_k1_PG_s110 - 1.01 s_k1_PG_s110  = 0
 w_k1_PG_s111_return_constraint: 0.945992 w_k1_PG_s11 - w_k1_PG_s111
                                 + 0.99 b_k1_PG_s111 - 1.01 s_k1_PG_s111  = 0
 w_k1_PG_s112_return_constraint: 0.871112 w_k1_PG_s11 - w_k1_PG_s112
                                 + 0.99 b_k1_PG_s112 - 1.01 s_k1_PG_s112  = 0
 w_k1_PG_s113_return_constraint: 0.830473 w_k1_PG_s11 - w_k1_PG_s113
                                 + 0.99 b_k1_PG_s113 - 1.01 s_k1_PG_s113  = 0
 w_k1_PG_s114_return_constraint: 1.06402 w_k1_PG_s11 - w_k1_PG_s114
                                 + 0.99 b_k1_PG_s114 - 1.01 s_k1_PG_s114  = 0
 w_k1_PG_s115_return_constraint: 1.01114 w_k1_PG_s11 - w_k1_PG_s115
                                 + 0.99 b_k1_PG_s115 - 1.01 s_k1_PG_s115  = 0
 w_k1_PG_s116_return_constraint: 1.14188 w_k1_PG_s11 - w_k1_PG_s116
                                 + 0.99 b_k1_PG_s116 - 1.01 s_k1_PG_s116  = 0
 w_k1_PG_s117_return_constraint: 0.851656 w_k1_PG_s11 - w_k1_PG_s117
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
 k1_wcvar_constraint:            wcvar - var_1 - 0.167000268309755 z_k1_s0000
                                 - 0.281999609207086 z_k1_s0001
                                 - 0.144000093769925 z_k1_s0002
                                 - 0.447000195339861 z_k1_s0003
                                 - 0.103000056434031 z_k1_s0004
                                 - 0.311000021679275 z_k1_s0005
                                 - 0.126000230973861 z_k1_s0006
                                 - 0.12200018581909 z_k1_s0007
                                 - 0.171000456629072 z_k1_s0010
                                 - 0.108000343213268 z_k1_s0011
                                 - 0.423000285390354 z_k1_s0012
                                 - 0.209000534957683 z_k1_s0013
                                 - 0.3840001556661 z_k1_s0014
                                 - 0.525000945131701 z_k1_s0015
                                 - 0.139000478373671 z_k1_s0016
                                 - 0.124000332340968 z_k1_s0017
                                 - 0.430000625643212 z_k1_s0020
                                 - 0.386001136890053 z_k1_s0021
                                 - 0.366001537093462 z_k1_s0022
                                 - 0.149000341290545 z_k1_s0023
                                 - 0.0900002296882909 z_k1_s0024
                                 - 0.116000263224849 z_k1_s0025
                                 - 0.0980002542072574 z_k1_s0026
                                 - 0.211000023661627 z_k1_s0027
                                 - 0.296001101842913 z_k1_s0030
                                 - 0.137000564628182 z_k1_s0031
                                 - 0.568002765538928 z_k1_s0032
                                 - 0.157000555352378 z_k1_s0033
                                 - 0.171000600489516 z_k1_s0034
                                 - 0.12700048321575 z_k1_s0035
                                 - 0.144000587196751 z_k1_s0036
                                 - 0.121000537628691 z_k1_s0037
                                 - 0.241000586583981 z_k1_s0040
                                 - 0.32400008374309 z_k1_s0041
                                 - 0.207000066430317 z_k1_s0042
                                 - 0.276999303110039 z_k1_s0043
                                 - 0.127000074481427 z_k1_s0044
                                 - 0.38500010669164 z_k1_s0045
                                 - 0.39599984379692 z_k1_s0046
                                 - 0.370000253638831 z_k1_s0047
                                 - 0.585002198027591 z_k1_s0050
                                 - 0.154000435302586 z_k1_s0051
                                 - 0.274999779180375 z_k1_s0052
                                 - 0.392002081991145 z_k1_s0053
                                 - 0.289001766649737 z_k1_s0054
                                 - 0.222000672884375 z_k1_s0055
                                 - 0.268001114452259 z_k1_s0056
                                 - 0.144000379814029 z_k1_s0057
                                 - 0.238000444692917 z_k1_s0060
                                 - 0.120999872906451 z_k1_s0061
                                 - 0.104000078142975 z_k1_s0062
                                 - 0.118000397066462 z_k1_s0063
                                 - 0.138000220670636 z_k1_s0064
                                 - 0.113000164665065 z_k1_s0065
                                 - 0.225000504050134 z_k1_s0066
                                 - 0.0490000655308614 z_k1_s0067
                                 - 0.626001712700637 z_k1_s0070
                                 - 0.826999994244414 z_k1_s0071
                                 - 0.362999938985095 z_k1_s0072
                                 - 0.312000287347034 z_k1_s0073
                                 - 0.367999328998053 z_k1_s0074
                                 - 0.496998793347613 z_k1_s0075
                                 - 0.252000351684263 z_k1_s0076
                                 - 0.210000293070219 z_k1_s0077
                                 - 0.576998864077394 z_k1_s0100
                                 - 0.28500002631735 z_k1_s0101
                                 - 0.959998709690209 z_k1_s0102
                                 - 0.237999897046085 z_k1_s0103
                                 - 0.385999432202528 z_k1_s0104
                                 - 0.200999845306997 z_k1_s0105
                                 - 0.282000103827683 z_k1_s0106
                                 - 0.429998967184218 z_k1_s0107
                                 - 1.1759993977639 z_k1_s0110
                                 - 0.594000573868589 z_k1_s0111
                                 - 0.324000141719014 z_k1_s0112
                                 - 0.911003672741551 z_k1_s0113
                                 - 0.493000709978939 z_k1_s0114
                                 - 1.45000208817335 z_k1_s0115
                                 - 0.528000370527458 z_k1_s0116
                                 - 0.804999101724672 z_k1_s0117
                                 - 1.28299515022068 z_k1_s0120
                                 - 0.948000787449274 z_k1_s0121
                                 - 0.900996244727558 z_k1_s0122
                                 - 0.532998991389424 z_k1_s0123
                                 - 1.16100015964311 z_k1_s0124
                                 - 0.828000010102206 z_k1_s0125
                                 - 0.645999277258533 z_k1_s0126
                                 - 0.39199941640476 z_k1_s0127
                                 - 0.864997811407738 z_k1_s0130
                                 - 0.387999952215555 z_k1_s0131
                                 - 0.645997582397232 z_k1_s0132
                                 - 0.556997844621165 z_k1_s0133
                                 - 0.408000062997766 z_k1_s0134
                                 - 0.739999979643048 z_k1_s0135
                                 - 0.182999846522583 z_k1_s0136
                                 - 0.789999112348924 z_k1_s0137
                                 - 0.358999878951979 z_k1_s0140
                                 - 0.252999884175082 z_k1_s0141
                                 - 0.523999739900966 z_k1_s0142
                                 - 0.57000031644169 z_k1_s0143
                                 - 0.215000106545554 z_k1_s0144
                                 - 0.696998573761763 z_k1_s0145
                                 - 0.801000194810248 z_k1_s0146
                                 - 0.233000078096357 z_k1_s0147
                                 - 2.53400252520366 z_k1_s0150
                                 - 1.17700096543143 z_k1_s0151
                                 - 0.906001686433732 z_k1_s0152
                                 - 2.66101027786024 z_k1_s0153
                                 - 1.00100091254554 z_k1_s0154
                                 - 3.535004836051 z_k1_s0155
                                 - 1.10600079314374 z_k1_s0156
                                 - 1.0630016144369 z_k1_s0157
                                 - 0.181000148919393 z_k1_s0160
                                 - 0.869001714345655 z_k1_s0161
                                 - 0.84500119684471 z_k1_s0162
                                 - 0.452999728329711 z_k1_s0163
                                 - 0.235999850758966 z_k1_s0164
                                 - 0.382000379889551 z_k1_s0165
                                 - 0.332000174762637 z_k1_s0166
                                 - 0.194000167332388 z_k1_s0167
                                 - 1.42100286309807 z_k1_s0170
                                 - 0.876001432678354 z_k1_s0171
                                 - 0.867999263256775 z_k1_s0172
                                 - 0.320000738986599 z_k1_s0173
                                 - 1.32100485588807 z_k1_s0174
                                 - 0.670998401983253 z_k1_s0175
                                 - 0.516000843906428 z_k1_s0176
                                 - 0.476000994121101 z_k1_s0177
                                 - 0.142999733631266 z_k1_s1000
                                 - 0.239000561174622 z_k1_s1001
                                 - 0.088000065680936 z_k1_s1002
                                 - 0.131999672121113 z_k1_s1003
                                 - 0.0760000954881076 z_k1_s1004
                                 - 0.0920000202118545 z_k1_s1005
                                 - 0.0830000958672696 z_k1_s1006
                                 - 0.212999737422887 z_k1_s1007
                                 - 0.220999608417662 z_k1_s1010
                                 - 0.194999763721414 z_k1_s1011
                                 - 0.266000268122265 z_k1_s1012
                                 - 0.230999548685449 z_k1_s1013
                                 - 0.14899985278968 z_k1_s1014
                                 - 0.26200029201515 z_k1_s1015
                                 - 0.377999599121484 z_k1_s1016
                                 - 0.155000002650265 z_k1_s1017
                                 - 0.129000324539418 z_k1_s1020
                                 - 0.484002025340419 z_k1_s1021
                                 - 0.230000640612281 z_k1_s1022
                                 - 0.16600028967369 z_k1_s1023
                                 - 0.173000485215766 z_k1_s1024
                                 - 0.565000861648121 z_k1_s1025
                                 - 0.574001592017589 z_k1_s1026
                                 - 0.258001164880003 z_k1_s1027
                                 - 0.790000158858782 z_k1_s1030
                                 - 0.387000178147431 z_k1_s1031
                                 - 0.611000860047607 z_k1_s1032
                                 - 0.232000238581608 z_k1_s1033
                                 - 0.434999585839192 z_k1_s1034
                                 - 0.511001232521782 z_k1_s1035
                                 - 0.164000216183623 z_k1_s1036
                                 - 0.316000408144042 z_k1_s1037
                                 - 0.294000312801723 z_k1_s1040
                                 - 0.256000107922503 z_k1_s1041
                                 - 0.136999501669173 z_k1_s1042
                                 - 0.103000095882909 z_k1_s1043
                                 - 0.0780000035044972 z_k1_s1044
                                 - 0.119000069053048 z_k1_s1045
                                 - 0.0960001076209737 z_k1_s1046
                                 - 0.259999564014756 z_k1_s1047
                                 - 0.0700000829700886 z_k1_s1050
                                 - 0.277000539661877 z_k1_s1051
                                 - 0.0890000550219123 z_k1_s1052
                                 - 0.208000461031365 z_k1_s1053
                                 - 0.0900000593614886 z_k1_s1054
                                 - 0.0940000767197938 z_k1_s1055
                                 - 0.204999785611881 z_k1_s1056
                                 - 0.0710000873096649 z_k1_s1057
                                 - 0.213000769455327 z_k1_s1060
                                 - 0.329000265909073 z_k1_s1061
                                 - 0.114000189822377 z_k1_s1062
                                 - 0.171000182383477 z_k1_s1063
                                 - 0.325999361310337 z_k1_s1064
                                 - 0.2909997250012 z_k1_s1065
                                 - 0.413999967517482 z_k1_s1066
                                 - 0.189000083071109 z_k1_s1067
                                 - 0.264000691428061 z_k1_s1070
                                 - 0.353001682845001 z_k1_s1071
                                 - 0.413000313165175 z_k1_s1072
                                 - 0.890003196150113 z_k1_s1073
                                 - 0.370001630670015 z_k1_s1074
                                 - 0.346001506740016 z_k1_s1075
                                 - 0.325000978425061 z_k1_s1076
                                 - 0.398002335089956 z_k1_s1077
                                 - 0.175999696301122 z_k1_s1100
                                 - 0.0609999108589023 z_k1_s1101
                                 - 0.149000468209046 z_k1_s1102
                                 - 0.249999944700236 z_k1_s1103
                                 - 0.104999953134113 z_k1_s1104
                                 - 0.0549999641940659 z_k1_s1105
                                 - 0.146999343388107 z_k1_s1106
                                 - 0.239000259181242 z_k1_s1107
                                 - 0.13399989735585 z_k1_s1110
                                 - 0.332999929637839 z_k1_s1111
                                 - 0.0909998976969202 z_k1_s1112
                                 - 0.0879999586694844 z_k1_s1113
                                 - 0.332999929637839 z_k1_s1114
                                 - 0.203999930661048 z_k1_s1115
                                 - 0.154999761747573 z_k1_s1116
                                 - 0.117999931343187 z_k1_s1117
                                 - 0.124999990630981 z_k1_s1120
                                 - 0.118999822300811 z_k1_s1121
                                 - 0.307999360955164 z_k1_s1122
                                 - 0.130999941461301 z_k1_s1123
                                 - 0.580998646233708 z_k1_s1124
                                 - 0.576000499707181 z_k1_s1125
                                 - 0.178999983103567 z_k1_s1126
                                 - 0.155999809087588 z_k1_s1127
                                 - 0.230999630706047 z_k1_s1130
                                 - 0.116999976475771 z_k1_s1131
                                 - 0.32899933612681 z_k1_s1132
                                 - 0.0669999657835633 z_k1_s1133
                                 - 0.326999930363051 z_k1_s1134
                                 - 0.13199991900344 z_k1_s1135
                                 - 0.216999239360274 z_k1_s1136
                                 - 0.0970000025388841 z_k1_s1137
                                 - 0.244000206497768 z_k1_s1140
                                 - 0.295000182773972 z_k1_s1141
                                 - 0.203999905104816 z_k1_s1142
                                 - 0.197999587895999 z_k1_s1143
                                 - 0.133999921666896 z_k1_s1144
                                 - 0.0899999981344588 z_k1_s1145
                                 - 0.106999990226527 z_k1_s1146
                                 - 0.0879999830648112 z_k1_s1147
                                 - 0.438997754252541 z_k1_s1150
                                 - 0.684998940618474 z_k1_s1151
                                 - 0.412998056351089 z_k1_s1152
                                 - 0.241999472215932 z_k1_s1153
                                 - 0.655000002961302 z_k1_s1154
                                 - 0.235999375385213 z_k1_s1155
                                 - 0.262999347174523 z_k1_s1156
                                 - 0.15999959225943 z_k1_s1157
                                 - 0.864000399241015 z_k1_s1160
                                 - 0.638997204611232 z_k1_s1161
                                 - 0.381999180636701 z_k1_s1162
                                 - 0.220999552669873 z_k1_s1163
                                 - 0.587996665934794 z_k1_s1164
                                 - 0.609999079947917 z_k1_s1165
                                 - 0.70499981065945 z_k1_s1166
                                 - 0.62699925950673 z_k1_s1167
                                 - 0.338001827982383 z_k1_s1170
                                 - 0.426999667356263 z_k1_s1171
                                 - 0.322001213058921 z_k1_s1172
                                 - 0.473001773511998 z_k1_s1173
                                 - 0.307001820445907 z_k1_s1174
                                 - 0.409002019824393 z_k1_s1175
                                 - 0.20100052023437 z_k1_s1176
                                 - 0.229000649248243 z_k1_s1177 >= 0
 z_k1_s0000_constraint:          var_1 + z_k1_s0000 + 0.959925 w_k1_ADM_s000
                                 + 1.01318 w_k1_AXP_s000 + 1.2659 w_k1_DIS_s000
                                 + 1.15165 w_k1_FDX_s000
                                 + 0.963259 w_k1_IBM_s000
                                 + 1.42352 w_k1_INTC_s000
                                 + 1.20449 w_k1_KO_s000 + 1.22988 w_k1_MAT_s000
                                 + 1.21171 w_k1_MSFT_s000 + 1.1295 w_k1_PG_s000
                                 >= 1
 z_k1_s0001_constraint:          var_1 + z_k1_s0001 + 0.918903 w_k1_ADM_s000
                                 + 0.890081 w_k1_AXP_s000
                                 + 1.23804 w_k1_DIS_s000
                                 + 0.969818 w_k1_FDX_s000
                                 + 1.24195 w_k1_IBM_s000
                                 + 1.10506 w_k1_INTC_s000
                                 + 1.19845 w_k1_KO_s000 + 1.26846 w_k1_MAT_s000
                                 + 1.13777 w_k1_MSFT_s000
                                 + 1.07375 w_k1_PG_s000 >= 1
 z_k1_s0002_constraint:          var_1 + z_k1_s0002 + 1.23795 w_k1_ADM_s000
                                 + 1.2433 w_k1_AXP_s000 + 1.66672 w_k1_DIS_s000
                                 + 1.33256 w_k1_FDX_s000
                                 + 1.44724 w_k1_IBM_s000
                                 + 1.36094 w_k1_INTC_s000
                                 + 1.43129 w_k1_KO_s000 + 1.66976 w_k1_MAT_s000
                                 + 1.36859 w_k1_MSFT_s000
                                 + 1.22338 w_k1_PG_s000 >= 1
 z_k1_s0003_constraint:          var_1 + z_k1_s0003 + 0.739733 w_k1_ADM_s000
                                 + 0.856073 w_k1_AXP_s000
                                 + 0.842145 w_k1_DIS_s000
                                 + 0.850858 w_k1_FDX_s000
                                 + 0.960361 w_k1_IBM_s000
                                 + 1.02007 w_k1_INTC_s000
                                 + 0.904283 w_k1_KO_s000
                                 + 1.40712 w_k1_MAT_s000
                                 + 0.890709 w_k1_MSFT_s000
                                 + 0.879492 w_k1_PG_s000 >= 1
 z_k1_s0004_constraint:          var_1 + z_k1_s0004 + 0.674763 w_k1_ADM_s000
                                 + 0.717771 w_k1_AXP_s000
                                 + 0.931182 w_k1_DIS_s000
                                 + 0.81685 w_k1_FDX_s000
                                 + 1.31013 w_k1_IBM_s000
                                 + 1.14686 w_k1_INTC_s000
                                 + 1.22734 w_k1_KO_s000 + 1.19997 w_k1_MAT_s000
                                 + 0.940906 w_k1_MSFT_s000
                                 + 1.05851 w_k1_PG_s000 >= 1
 z_k1_s0005_constraint:          var_1 + z_k1_s0005 + 1.17766 w_k1_ADM_s000
                                 + 1.20937 w_k1_AXP_s000
                                 + 1.22398 w_k1_DIS_s000
                                 + 1.33327 w_k1_FDX_s000
                                 + 0.809597 w_k1_IBM_s000
                                 + 1.01537 w_k1_INTC_s000
                                 + 0.931069 w_k1_KO_s000
                                 + 1.39763 w_k1_MAT_s000
                                 + 1.05945 w_k1_MSFT_s000
                                 + 1.14705 w_k1_PG_s000 >= 1
 z_k1_s0006_constraint:          var_1 + z_k1_s0006 + 0.673692 w_k1_ADM_s000
                                 + 0.631858 w_k1_AXP_s000
                                 + 0.61663 w_k1_DIS_s000
                                 + 0.740478 w_k1_FDX_s000
                                 + 0.761559 w_k1_IBM_s000
                                 + 0.727514 w_k1_INTC_s000
                                 + 0.751587 w_k1_KO_s000
                                 + 0.829168 w_k1_MAT_s000
                                 + 0.719669 w_k1_MSFT_s000
                                 + 0.84641 w_k1_PG_s000 >= 1
 z_k1_s0007_constraint:          var_1 + z_k1_s0007 + 0.813788 w_k1_ADM_s000
                                 + 0.669881 w_k1_AXP_s000
                                 + 1.02733 w_k1_DIS_s000
                                 + 0.909408 w_k1_FDX_s000
                                 + 0.78757 w_k1_IBM_s000
                                 + 0.942101 w_k1_INTC_s000
                                 + 1.01675 w_k1_KO_s000
                                 + 0.913633 w_k1_MAT_s000
                                 + 1.10312 w_k1_MSFT_s000
                                 + 1.04188 w_k1_PG_s000 >= 1
 z_k1_s0010_constraint:          var_1 + z_k1_s0010 + 0.878057 w_k1_ADM_s001
                                 + 1.07086 w_k1_AXP_s001
                                 + 0.603582 w_k1_DIS_s001
                                 + 0.90634 w_k1_FDX_s001
                                 + 0.735963 w_k1_IBM_s001
                                 + 0.980508 w_k1_INTC_s001
                                 + 0.660985 w_k1_KO_s001
                                 + 0.886623 w_k1_MAT_s001
                                 + 0.886969 w_k1_MSFT_s001
                                 + 0.884842 w_k1_PG_s001 >= 1
 z_k1_s0011_constraint:          var_1 + z_k1_s0011 + 1.18323 w_k1_ADM_s001
                                 + 1.46609 w_k1_AXP_s001
                                 + 1.51625 w_k1_DIS_s001
                                 + 1.05757 w_k1_FDX_s001
                                 + 1.93131 w_k1_IBM_s001
                                 + 1.47385 w_k1_INTC_s001
                                 + 1.42617 w_k1_KO_s001 + 1.52653 w_k1_MAT_s001
                                 + 1.35394 w_k1_MSFT_s001
                                 + 1.31966 w_k1_PG_s001 >= 1
 z_k1_s0012_constraint:          var_1 + z_k1_s0012 + 1.07011 w_k1_ADM_s001
                                 + 1.24018 w_k1_AXP_s001
                                 + 1.20298 w_k1_DIS_s001
                                 + 1.12351 w_k1_FDX_s001
                                 + 1.31072 w_k1_IBM_s001
                                 + 1.29448 w_k1_INTC_s001 + 1.0938 w_k1_KO_s001
                                 + 1.31434 w_k1_MAT_s001
                                 + 1.19342 w_k1_MSFT_s001
                                 + 1.15777 w_k1_PG_s001 >= 1
 z_k1_s0013_constraint:          var_1 + z_k1_s0013 + 1.32916 w_k1_ADM_s001
                                 + 1.5637 w_k1_AXP_s001 + 1.74348 w_k1_DIS_s001
                                 + 1.42762 w_k1_FDX_s001
                                 + 0.989535 w_k1_IBM_s001
                                 + 1.36123 w_k1_INTC_s001
                                 + 1.17776 w_k1_KO_s001 + 1.07165 w_k1_MAT_s001
                                 + 1.44777 w_k1_MSFT_s001
                                 + 1.27521 w_k1_PG_s001 >= 1
 z_k1_s0014_constraint:          var_1 + z_k1_s0014 + 0.865199 w_k1_ADM_s001
                                 + 0.816298 w_k1_AXP_s001
                                 + 0.937191 w_k1_DIS_s001
                                 + 1.0139 w_k1_FDX_s001
                                 + 0.697879 w_k1_IBM_s001
                                 + 1.08183 w_k1_INTC_s001
                                 + 0.793022 w_k1_KO_s001
                                 + 0.937177 w_k1_MAT_s001
                                 + 1.1417 w_k1_MSFT_s001
                                 + 0.947464 w_k1_PG_s001 >= 1
 z_k1_s0015_constraint:          var_1 + z_k1_s0015 + 0.977389 w_k1_ADM_s001
                                 + 1.08662 w_k1_AXP_s001
                                 + 0.934083 w_k1_DIS_s001
                                 + 0.993678 w_k1_FDX_s001
                                 + 0.954361 w_k1_IBM_s001
                                 + 1.52135 w_k1_INTC_s001
                                 + 0.905178 w_k1_KO_s001
                                 + 0.792383 w_k1_MAT_s001
                                 + 1.21564 w_k1_MSFT_s001
                                 + 0.981623 w_k1_PG_s001 >= 1
 z_k1_s0016_constraint:          var_1 + z_k1_s0016 + 0.879317 w_k1_ADM_s001
                                 + 0.983103 w_k1_AXP_s001
                                 + 0.650976 w_k1_DIS_s001
                                 + 0.785741 w_k1_FDX_s001
                                 + 1.32606 w_k1_IBM_s001
                                 + 1.27409 w_k1_INTC_s001
                                 + 1.00471 w_k1_KO_s001 + 1.48452 w_k1_MAT_s001
                                 + 0.94656 w_k1_MSFT_s001
                                 + 0.976974 w_k1_PG_s001 >= 1
 z_k1_s0017_constraint:          var_1 + z_k1_s0017 + 0.648655 w_k1_ADM_s001
                                 + 0.617853 w_k1_AXP_s001
                                 + 0.572676 w_k1_DIS_s001
                                 + 0.719852 w_k1_FDX_s001
                                 + 0.651626 w_k1_IBM_s001
                                 + 0.768513 w_k1_INTC_s001
                                 + 0.567321 w_k1_KO_s001 + 0.7063 w_k1_MAT_s001
                                 + 0.749179 w_k1_MSFT_s001
                                 + 0.806799 w_k1_PG_s001 >= 1
 z_k1_s0020_constraint:          var_1 + z_k1_s0020 + 0.958513 w_k1_ADM_s002
                                 + 0.86086 w_k1_AXP_s002
                                 + 0.692752 w_k1_DIS_s002
                                 + 0.846385 w_k1_FDX_s002
                                 + 0.927908 w_k1_IBM_s002
                                 + 0.833074 w_k1_INTC_s002
                                 + 0.767293 w_k1_KO_s002
                                 + 0.980338 w_k1_MAT_s002
                                 + 0.930978 w_k1_MSFT_s002
                                 + 0.867265 w_k1_PG_s002 >= 1
 z_k1_s0021_constraint:          var_1 + z_k1_s0021 + 1.10065 w_k1_ADM_s002
                                 + 1.31889 w_k1_AXP_s002
                                 + 0.939797 w_k1_DIS_s002
                                 + 0.970791 w_k1_FDX_s002
                                 + 1.34793 w_k1_IBM_s002
                                 + 0.990789 w_k1_INTC_s002
                                 + 1.0242 w_k1_KO_s002 + 1.4581 w_k1_MAT_s002
                                 + 1.036 w_k1_MSFT_s002 + 0.956531 w_k1_PG_s002
                                 >= 1
 z_k1_s0022_constraint:          var_1 + z_k1_s0022 + 1.23199 w_k1_ADM_s002
                                 + 1.37933 w_k1_AXP_s002
                                 + 1.37945 w_k1_DIS_s002
                                 + 1.30361 w_k1_FDX_s002
                                 + 1.11699 w_k1_IBM_s002
                                 + 0.881932 w_k1_INTC_s002
                                 + 1.15228 w_k1_KO_s002 + 1.13835 w_k1_MAT_s002
                                 + 1.17927 w_k1_MSFT_s002
                                 + 1.06636 w_k1_PG_s002 >= 1
 z_k1_s0023_constraint:          var_1 + z_k1_s0023 + 1.09454 w_k1_ADM_s002
                                 + 0.859376 w_k1_AXP_s002
                                 + 1.22756 w_k1_DIS_s002
                                 + 1.04681 w_k1_FDX_s002
                                 + 0.877738 w_k1_IBM_s002
                                 + 0.950888 w_k1_INTC_s002
                                 + 1.14191 w_k1_KO_s002 + 0.74545 w_k1_MAT_s002
                                 + 1.14475 w_k1_MSFT_s002
                                 + 1.05615 w_k1_PG_s002 >= 1
 z_k1_s0024_constraint:          var_1 + z_k1_s0024 + 0.83468 w_k1_ADM_s002
                                 + 0.660852 w_k1_AXP_s002
                                 + 0.560677 w_k1_DIS_s002
                                 + 0.704408 w_k1_FDX_s002
                                 + 0.568851 w_k1_IBM_s002
                                 + 0.737025 w_k1_INTC_s002
                                 + 0.630336 w_k1_KO_s002
                                 + 0.447958 w_k1_MAT_s002
                                 + 0.629947 w_k1_MSFT_s002
                                 + 0.662587 w_k1_PG_s002 >= 1
 z_k1_s0025_constraint:          var_1 + z_k1_s0025 + 0.970247 w_k1_ADM_s002
                                 + 0.830208 w_k1_AXP_s002
                                 + 0.732744 w_k1_DIS_s002
                                 + 0.870255 w_k1_FDX_s002
                                 + 0.686106 w_k1_IBM_s002
                                 + 0.88742 w_k1_INTC_s002
                                 + 0.860222 w_k1_KO_s002
                                 + 0.711382 w_k1_MAT_s002
                                 + 1.03158 w_k1_MSFT_s002
                                 + 0.914849 w_k1_PG_s002 >= 1
 z_k1_s0026_constraint:          var_1 + z_k1_s0026 + 1.48379 w_k1_ADM_s002
                                 + 1.19124 w_k1_AXP_s002
                                 + 2.17092 w_k1_DIS_s002
                                 + 1.47541 w_k1_FDX_s002
                                 + 1.41175 w_k1_IBM_s002
                                 + 1.15541 w_k1_INTC_s002
                                 + 1.72784 w_k1_KO_s002 + 1.3144 w_k1_MAT_s002
                                 + 1.54657 w_k1_MSFT_s002
                                 + 1.29168 w_k1_PG_s002 >= 1
 z_k1_s0027_constraint:          var_1 + z_k1_s0027 + 1.20183 w_k1_ADM_s002
                                 + 0.938974 w_k1_AXP_s002
                                 + 1.07381 w_k1_DIS_s002
                                 + 1.13257 w_k1_FDX_s002
                                 + 0.785701 w_k1_IBM_s002
                                 + 0.976374 w_k1_INTC_s002
                                 + 0.956284 w_k1_KO_s002
                                 + 0.886267 w_k1_MAT_s002
                                 + 1.1835 w_k1_MSFT_s002
                                 + 0.876436 w_k1_PG_s002 >= 1
 z_k1_s0030_constraint:          var_1 + z_k1_s0030 + 0.869652 w_k1_ADM_s003
                                 + 0.789692 w_k1_AXP_s003
                                 + 0.852094 w_k1_DIS_s003
                                 + 0.856543 w_k1_FDX_s003
                                 + 1.04398 w_k1_IBM_s003
                                 + 0.841833 w_k1_INTC_s003
                                 + 0.883596 w_k1_KO_s003
                                 + 0.87264 w_k1_MAT_s003
                                 + 0.848685 w_k1_MSFT_s003
                                 + 0.886786 w_k1_PG_s003 >= 1
 z_k1_s0031_constraint:          var_1 + z_k1_s0031 + 0.882792 w_k1_ADM_s003
                                 + 1.14111 w_k1_AXP_s003
                                 + 0.813443 w_k1_DIS_s003
                                 + 1.00082 w_k1_FDX_s003
                                 + 0.973043 w_k1_IBM_s003
                                 + 0.974986 w_k1_INTC_s003
                                 + 0.841406 w_k1_KO_s003
                                 + 1.10043 w_k1_MAT_s003
                                 + 0.838483 w_k1_MSFT_s003
                                 + 0.919368 w_k1_PG_s003 >= 1
 z_k1_s0032_constraint:          var_1 + z_k1_s0032 + 0.969349 w_k1_ADM_s003
                                 + 1.12182 w_k1_AXP_s003 + 1.1255 w_k1_DIS_s003
                                 + 1.03649 w_k1_FDX_s003 + 1.1385 w_k1_IBM_s003
                                 + 0.970835 w_k1_INTC_s003
                                 + 1.15074 w_k1_KO_s003 + 1.26658 w_k1_MAT_s003
                                 + 1.03106 w_k1_MSFT_s003
                                 + 1.11564 w_k1_PG_s003 >= 1
 z_k1_s0033_constraint:          var_1 + z_k1_s0033 + 0.923675 w_k1_ADM_s003
                                 + 0.82879 w_k1_AXP_s003
                                 + 0.973222 w_k1_DIS_s003
                                 + 1.10912 w_k1_FDX_s003
                                 + 0.788693 w_k1_IBM_s003
                                 + 1.14292 w_k1_INTC_s003
                                 + 1.07289 w_k1_KO_s003 + 1.1028 w_k1_MAT_s003
                                 + 1.1278 w_k1_MSFT_s003 + 1.10964 w_k1_PG_s003
                                 >= 1
 z_k1_s0034_constraint:          var_1 + z_k1_s0034 + 0.661335 w_k1_ADM_s003
                                 + 0.66229 w_k1_AXP_s003
                                 + 0.659358 w_k1_DIS_s003
                                 + 0.765459 w_k1_FDX_s003
                                 + 0.874106 w_k1_IBM_s003
                                 + 0.799036 w_k1_INTC_s003
                                 + 0.849543 w_k1_KO_s003
                                 + 0.848644 w_k1_MAT_s003
                                 + 0.723489 w_k1_MSFT_s003
                                 + 0.805822 w_k1_PG_s003 >= 1
 z_k1_s0035_constraint:          var_1 + z_k1_s0035 + 0.658109 w_k1_ADM_s003
                                 + 0.792476 w_k1_AXP_s003
                                 + 0.706328 w_k1_DIS_s003
                                 + 0.901431 w_k1_FDX_s003
                                 + 0.693997 w_k1_IBM_s003
                                 + 1.04534 w_k1_INTC_s003
                                 + 0.790906 w_k1_KO_s003
                                 + 0.706837 w_k1_MAT_s003
                                 + 0.990607 w_k1_MSFT_s003
                                 + 0.921879 w_k1_PG_s003 >= 1
 z_k1_s0036_constraint:          var_1 + z_k1_s0036 + 0.731139 w_k1_ADM_s003
                                 + 0.765458 w_k1_AXP_s003
                                 + 1.22562 w_k1_DIS_s003
                                 + 1.00212 w_k1_FDX_s003
                                 + 1.07819 w_k1_IBM_s003
                                 + 1.25662 w_k1_INTC_s003
                                 + 1.26761 w_k1_KO_s003 + 1.17907 w_k1_MAT_s003
                                 + 1.2386 w_k1_MSFT_s003 + 1.15135 w_k1_PG_s003
                                 >= 1
 z_k1_s0037_constraint:          var_1 + z_k1_s0037 + 0.885783 w_k1_ADM_s003
                                 + 1.44105 w_k1_AXP_s003
                                 + 1.76318 w_k1_DIS_s003 + 1.386 w_k1_FDX_s003
                                 + 1.45308 w_k1_IBM_s003
                                 + 1.16606 w_k1_INTC_s003
                                 + 1.52763 w_k1_KO_s003 + 1.92779 w_k1_MAT_s003
                                 + 1.35959 w_k1_MSFT_s003
                                 + 1.23431 w_k1_PG_s003 >= 1
 z_k1_s0040_constraint:          var_1 + z_k1_s0040 + 0.735717 w_k1_ADM_s004
                                 + 0.7259 w_k1_AXP_s004 + 0.66862 w_k1_DIS_s004
                                 + 0.737576 w_k1_FDX_s004
                                 + 0.831473 w_k1_IBM_s004
                                 + 0.725023 w_k1_INTC_s004
                                 + 0.793106 w_k1_KO_s004
                                 + 0.955664 w_k1_MAT_s004
                                 + 0.771675 w_k1_MSFT_s004
                                 + 0.774484 w_k1_PG_s004 >= 1
 z_k1_s0041_constraint:          var_1 + z_k1_s0041 + 0.762107 w_k1_ADM_s004
                                 + 0.746688 w_k1_AXP_s004
                                 + 0.753213 w_k1_DIS_s004
                                 + 0.795762 w_k1_FDX_s004
                                 + 0.656151 w_k1_IBM_s004
                                 + 0.859238 w_k1_INTC_s004
                                 + 0.79009 w_k1_KO_s004
                                 + 0.755604 w_k1_MAT_s004
                                 + 0.880847 w_k1_MSFT_s004
                                 + 0.900432 w_k1_PG_s004 >= 1
 z_k1_s0042_constraint:          var_1 + z_k1_s0042 + 1.34481 w_k1_ADM_s004
                                 + 1.49965 w_k1_AXP_s004
                                 + 1.77204 w_k1_DIS_s004
                                 + 1.33082 w_k1_FDX_s004
                                 + 1.62052 w_k1_IBM_s004
                                 + 1.46963 w_k1_INTC_s004
                                 + 1.62374 w_k1_KO_s004 + 1.45222 w_k1_MAT_s004
                                 + 1.36702 w_k1_MSFT_s004
                                 + 1.45269 w_k1_PG_s004 >= 1
 z_k1_s0043_constraint:          var_1 + z_k1_s0043 + 0.838389 w_k1_ADM_s004
                                 + 0.765971 w_k1_AXP_s004
                                 + 1.01549 w_k1_DIS_s004
                                 + 0.884981 w_k1_FDX_s004
                                 + 1.11989 w_k1_IBM_s004
                                 + 1.00684 w_k1_INTC_s004 + 1.1402 w_k1_KO_s004
                                 + 1.06931 w_k1_MAT_s004
                                 + 1.04498 w_k1_MSFT_s004
                                 + 1.11467 w_k1_PG_s004 >= 1
 z_k1_s0044_constraint:          var_1 + z_k1_s0044 + 0.858715 w_k1_ADM_s004
                                 + 0.979156 w_k1_AXP_s004
                                 + 0.701224 w_k1_DIS_s004
                                 + 0.785718 w_k1_FDX_s004
                                 + 0.992076 w_k1_IBM_s004
                                 + 0.86874 w_k1_INTC_s004
                                 + 0.800808 w_k1_KO_s004
                                 + 1.30286 w_k1_MAT_s004
                                 + 0.964651 w_k1_MSFT_s004
                                 + 0.981695 w_k1_PG_s004 >= 1
 z_k1_s0045_constraint:          var_1 + z_k1_s0045 + 1.22959 w_k1_ADM_s004
                                 + 1.0417 w_k1_AXP_s004 + 1.12596 w_k1_DIS_s004
                                 + 1.10457 w_k1_FDX_s004
                                 + 0.837972 w_k1_IBM_s004
                                 + 1.02085 w_k1_INTC_s004
                                 + 0.952608 w_k1_KO_s004
                                 + 1.19728 w_k1_MAT_s004
                                 + 1.08778 w_k1_MSFT_s004
                                 + 1.04698 w_k1_PG_s004 >= 1
 z_k1_s0046_constraint:          var_1 + z_k1_s0046 + 1.03243 w_k1_ADM_s004
                                 + 1.06194 w_k1_AXP_s004
                                 + 1.31352 w_k1_DIS_s004
                                 + 1.17955 w_k1_FDX_s004
                                 + 1.19032 w_k1_IBM_s004
                                 + 1.11164 w_k1_INTC_s004
                                 + 1.10642 w_k1_KO_s004 + 1.44369 w_k1_MAT_s004
                                 + 1.20502 w_k1_MSFT_s004
                                 + 1.21748 w_k1_PG_s004 >= 1
 z_k1_s0047_constraint:          var_1 + z_k1_s0047 + 1.1468 w_k1_ADM_s004
                                 + 1.24015 w_k1_AXP_s004
                                 + 1.12953 w_k1_DIS_s004
                                 + 1.02753 w_k1_FDX_s004
                                 + 1.19521 w_k1_IBM_s004
                                 + 0.976897 w_k1_INTC_s004
                                 + 1.26152 w_k1_KO_s004 + 1.25305 w_k1_MAT_s004
                                 + 1.07787 w_k1_MSFT_s004
                                 + 1.15421 w_k1_PG_s004 >= 1
 z_k1_s0050_constraint:          var_1 + z_k1_s0050 + 1.10809 w_k1_ADM_s005
                                 + 1.0872 w_k1_AXP_s005 + 1.00873 w_k1_DIS_s005
                                 + 0.948222 w_k1_FDX_s005
                                 + 1.34803 w_k1_IBM_s005
                                 + 1.08082 w_k1_INTC_s005
                                 + 1.19578 w_k1_KO_s005 + 1.17939 w_k1_MAT_s005
                                 + 0.994915 w_k1_MSFT_s005
                                 + 1.06998 w_k1_PG_s005 >= 1
 z_k1_s0051_constraint:          var_1 + z_k1_s0051 + 0.941489 w_k1_ADM_s005
                                 + 0.785636 w_k1_AXP_s005
                                 + 1.04401 w_k1_DIS_s005
                                 + 1.03219 w_k1_FDX_s005
                                 + 0.812808 w_k1_IBM_s005
                                 + 0.980696 w_k1_INTC_s005
                                 + 1.06392 w_k1_KO_s005
                                 + 0.685578 w_k1_MAT_s005
                                 + 1.11068 w_k1_MSFT_s005
                                 + 0.899174 w_k1_PG_s005 >= 1
 z_k1_s0052_constraint:          var_1 + z_k1_s0052 + 1.06328 w_k1_ADM_s005
                                 + 0.818664 w_k1_AXP_s005
                                 + 0.759777 w_k1_DIS_s005
                                 + 0.889988 w_k1_FDX_s005
                                 + 0.766481 w_k1_IBM_s005
                                 + 1.00786 w_k1_INTC_s005
                                 + 0.862945 w_k1_KO_s005
                                 + 0.87536 w_k1_MAT_s005
                                 + 1.03235 w_k1_MSFT_s005
                                 + 0.925625 w_k1_PG_s005 >= 1
 z_k1_s0053_constraint:          var_1 + z_k1_s0053 + 0.903264 w_k1_ADM_s005
                                 + 1.0226 w_k1_AXP_s005 + 1.14867 w_k1_DIS_s005
                                 + 1.09593 w_k1_FDX_s005
                                 + 0.956803 w_k1_IBM_s005
                                 + 1.16375 w_k1_INTC_s005
                                 + 1.07683 w_k1_KO_s005
                                 + 0.908083 w_k1_MAT_s005
                                 + 1.15286 w_k1_MSFT_s005
                                 + 0.946378 w_k1_PG_s005 >= 1
 z_k1_s0054_constraint:          var_1 + z_k1_s0054 + 1.11237 w_k1_ADM_s005
                                 + 1.081 w_k1_AXP_s005 + 1.59116 w_k1_DIS_s005
                                 + 1.10431 w_k1_FDX_s005
                                 + 1.26048 w_k1_IBM_s005
                                 + 1.42087 w_k1_INTC_s005
                                 + 1.44825 w_k1_KO_s005 + 1.25284 w_k1_MAT_s005
                                 + 1.48183 w_k1_MSFT_s005
                                 + 1.16263 w_k1_PG_s005 >= 1
 z_k1_s0055_constraint:          var_1 + z_k1_s0055 + 1.39154 w_k1_ADM_s005
                                 + 1.75997 w_k1_AXP_s005
                                 + 1.48872 w_k1_DIS_s005
                                 + 1.33592 w_k1_FDX_s005
                                 + 1.36858 w_k1_IBM_s005
                                 + 1.73409 w_k1_INTC_s005
                                 + 1.32108 w_k1_KO_s005 + 1.0527 w_k1_MAT_s005
                                 + 1.47621 w_k1_MSFT_s005
                                 + 1.23986 w_k1_PG_s005 >= 1
 z_k1_s0056_constraint:          var_1 + z_k1_s0056 + 0.759398 w_k1_ADM_s005
                                 + 0.668951 w_k1_AXP_s005
                                 + 0.6667 w_k1_DIS_s005
                                 + 0.735598 w_k1_FDX_s005
                                 + 0.864798 w_k1_IBM_s005
                                 + 0.884265 w_k1_INTC_s005
                                 + 0.65353 w_k1_KO_s005
                                 + 0.564946 w_k1_MAT_s005
                                 + 0.806764 w_k1_MSFT_s005
                                 + 0.759479 w_k1_PG_s005 >= 1
 z_k1_s0057_constraint:          var_1 + z_k1_s0057 + 1.25141 w_k1_ADM_s005
                                 + 1.25083 w_k1_AXP_s005
                                 + 1.02152 w_k1_DIS_s005 + 1.2514 w_k1_FDX_s005
                                 + 0.834174 w_k1_IBM_s005
                                 + 1.14964 w_k1_INTC_s005
                                 + 0.916373 w_k1_KO_s005
                                 + 0.774352 w_k1_MAT_s005
                                 + 1.22249 w_k1_MSFT_s005
                                 + 1.01129 w_k1_PG_s005 >= 1
 z_k1_s0060_constraint:          var_1 + z_k1_s0060 + 0.877511 w_k1_ADM_s006
                                 + 1.19351 w_k1_AXP_s006
                                 + 1.07153 w_k1_DIS_s006
                                 + 0.886387 w_k1_FDX_s006
                                 + 1.43984 w_k1_IBM_s006
                                 + 0.763192 w_k1_INTC_s006
                                 + 0.961151 w_k1_KO_s006 + 1.4114 w_k1_MAT_s006
                                 + 0.889076 w_k1_MSFT_s006
                                 + 1.10038 w_k1_PG_s006 >= 1
 z_k1_s0061_constraint:          var_1 + z_k1_s0061 + 1.07049 w_k1_ADM_s006
                                 + 0.755385 w_k1_AXP_s006
                                 + 0.991538 w_k1_DIS_s006
                                 + 1.04231 w_k1_FDX_s006
                                 + 0.656702 w_k1_IBM_s006
                                 + 0.960929 w_k1_INTC_s006
                                 + 0.821357 w_k1_KO_s006
                                 + 0.562864 w_k1_MAT_s006
                                 + 1.12912 w_k1_MSFT_s006
                                 + 0.934031 w_k1_PG_s006 >= 1
 z_k1_s0062_constraint:          var_1 + z_k1_s0062 + 0.817742 w_k1_ADM_s006
                                 + 1.10816 w_k1_AXP_s006
                                 + 0.878003 w_k1_DIS_s006
                                 + 1.04711 w_k1_FDX_s006
                                 + 0.828934 w_k1_IBM_s006
                                 + 1.04874 w_k1_INTC_s006
                                 + 0.750192 w_k1_KO_s006
                                 + 1.00157 w_k1_MAT_s006
                                 + 0.966145 w_k1_MSFT_s006
                                 + 1.07222 w_k1_PG_s006 >= 1
 z_k1_s0063_constraint:          var_1 + z_k1_s0063 + 0.951375 w_k1_ADM_s006
                                 + 0.904024 w_k1_AXP_s006
                                 + 0.835622 w_k1_DIS_s006
                                 + 0.902188 w_k1_FDX_s006
                                 + 1.01119 w_k1_IBM_s006
                                 + 0.794678 w_k1_INTC_s006
                                 + 0.896204 w_k1_KO_s006
                                 + 0.908749 w_k1_MAT_s006
                                 + 0.857613 w_k1_MSFT_s006
                                 + 1.06681 w_k1_PG_s006 >= 1
 z_k1_s0064_constraint:          var_1 + z_k1_s0064 + 0.982446 w_k1_ADM_s006
                                 + 0.567513 w_k1_AXP_s006
                                 + 0.774351 w_k1_DIS_s006
                                 + 0.870511 w_k1_FDX_s006
                                 + 0.789292 w_k1_IBM_s006
                                 + 0.887376 w_k1_INTC_s006
                                 + 0.717033 w_k1_KO_s006
                                 + 0.472213 w_k1_MAT_s006
                                 + 0.765048 w_k1_MSFT_s006
                                 + 0.820465 w_k1_PG_s006 >= 1
 z_k1_s0065_constraint:          var_1 + z_k1_s0065 + 0.608396 w_k1_ADM_s006
                                 + 0.77731 w_k1_AXP_s006
                                 + 0.612242 w_k1_DIS_s006
                                 + 0.846034 w_k1_FDX_s006
                                 + 0.767396 w_k1_IBM_s006
                                 + 0.977201 w_k1_INTC_s006
                                 + 0.626371 w_k1_KO_s006
                                 + 0.801079 w_k1_MAT_s006
                                 + 0.800416 w_k1_MSFT_s006
                                 + 0.90342 w_k1_PG_s006 >= 1
 z_k1_s0066_constraint:          var_1 + z_k1_s0066 + 1.1773 w_k1_ADM_s006
                                 + 1.00159 w_k1_AXP_s006
                                 + 1.37954 w_k1_DIS_s006
                                 + 1.15985 w_k1_FDX_s006
                                 + 1.02645 w_k1_IBM_s006
                                 + 1.0545 w_k1_INTC_s006 + 1.1928 w_k1_KO_s006
                                 + 1.12621 w_k1_MAT_s006
                                 + 1.26401 w_k1_MSFT_s006 + 1.1388 w_k1_PG_s006
                                 >= 1
 z_k1_s0067_constraint:          var_1 + z_k1_s0067 + 1.09768 w_k1_ADM_s006
                                 + 1.28283 w_k1_AXP_s006
                                 + 1.17196 w_k1_DIS_s006
                                 + 1.25506 w_k1_FDX_s006
                                 + 0.793299 w_k1_IBM_s006
                                 + 1.50219 w_k1_INTC_s006
                                 + 0.900895 w_k1_KO_s006
                                 + 0.77731 w_k1_MAT_s006
                                 + 1.03458 w_k1_MSFT_s006
                                 + 1.10558 w_k1_PG_s006 >= 1
 z_k1_s0070_constraint:          var_1 + z_k1_s0070 + 0.917843 w_k1_ADM_s007
                                 + 0.943909 w_k1_AXP_s007
                                 + 0.827909 w_k1_DIS_s007
                                 + 0.928407 w_k1_FDX_s007
                                 + 0.874534 w_k1_IBM_s007
                                 + 0.848791 w_k1_INTC_s007
                                 + 0.841193 w_k1_KO_s007
                                 + 1.14096 w_k1_MAT_s007
                                 + 0.852521 w_k1_MSFT_s007
                                 + 0.909858 w_k1_PG_s007 >= 1
 z_k1_s0071_constraint:          var_1 + z_k1_s0071 + 1.20729 w_k1_ADM_s007
                                 + 1.305 w_k1_AXP_s007 + 1.15189 w_k1_DIS_s007
                                 + 1.09925 w_k1_FDX_s007
                                 + 1.34501 w_k1_IBM_s007
                                 + 1.0435 w_k1_INTC_s007 + 1.41862 w_k1_KO_s007
                                 + 1.42173 w_k1_MAT_s007
                                 + 1.10028 w_k1_MSFT_s007
                                 + 1.17787 w_k1_PG_s007 >= 1
 z_k1_s0072_constraint:          var_1 + z_k1_s0072 + 0.729881 w_k1_ADM_s007
                                 + 0.781157 w_k1_AXP_s007
                                 + 0.6549 w_k1_DIS_s007
                                 + 0.648218 w_k1_FDX_s007
                                 + 1.01145 w_k1_IBM_s007
                                 + 0.723686 w_k1_INTC_s007
                                 + 0.957493 w_k1_KO_s007
                                 + 0.858744 w_k1_MAT_s007
                                 + 0.753908 w_k1_MSFT_s007
                                 + 0.860323 w_k1_PG_s007 >= 1
 z_k1_s0073_constraint:          var_1 + z_k1_s0073 + 0.961165 w_k1_ADM_s007
                                 + 1.26304 w_k1_AXP_s007
                                 + 1.88118 w_k1_DIS_s007
                                 + 1.35925 w_k1_FDX_s007
                                 + 1.06989 w_k1_IBM_s007
                                 + 1.07991 w_k1_INTC_s007
                                 + 1.48625 w_k1_KO_s007 + 1.65157 w_k1_MAT_s007
                                 + 1.3011 w_k1_MSFT_s007 + 1.1656 w_k1_PG_s007
                                 >= 1
 z_k1_s0074_constraint:          var_1 + z_k1_s0074 + 1.06951 w_k1_ADM_s007
                                 + 1.48267 w_k1_AXP_s007
                                 + 1.18072 w_k1_DIS_s007
                                 + 1.33012 w_k1_FDX_s007
                                 + 0.767635 w_k1_IBM_s007
                                 + 1.07255 w_k1_INTC_s007
                                 + 0.974812 w_k1_KO_s007
                                 + 0.829426 w_k1_MAT_s007
                                 + 1.13094 w_k1_MSFT_s007
                                 + 1.08682 w_k1_PG_s007 >= 1
 z_k1_s0075_constraint:          var_1 + z_k1_s0075 + 0.85065 w_k1_ADM_s007
                                 + 1.04655 w_k1_AXP_s007
                                 + 0.889652 w_k1_DIS_s007
                                 + 0.821854 w_k1_FDX_s007
                                 + 1.23297 w_k1_IBM_s007
                                 + 0.992087 w_k1_INTC_s007
                                 + 1.03382 w_k1_KO_s007 + 1.43952 w_k1_MAT_s007
                                 + 0.891988 w_k1_MSFT_s007
                                 + 0.987527 w_k1_PG_s007 >= 1
 z_k1_s0076_constraint:          var_1 + z_k1_s0076 + 0.81278 w_k1_ADM_s007
                                 + 1.03884 w_k1_AXP_s007
                                 + 1.34067 w_k1_DIS_s007
                                 + 1.11097 w_k1_FDX_s007
                                 + 0.748613 w_k1_IBM_s007
                                 + 1.0847 w_k1_INTC_s007 + 1.0343 w_k1_KO_s007
                                 + 0.668268 w_k1_MAT_s007
                                 + 1.14228 w_k1_MSFT_s007
                                 + 0.973955 w_k1_PG_s007 >= 1
 z_k1_s0077_constraint:          var_1 + z_k1_s0077 + 0.690008 w_k1_ADM_s007
                                 + 0.664507 w_k1_AXP_s007
                                 + 0.683576 w_k1_DIS_s007
                                 + 0.745685 w_k1_FDX_s007
                                 + 0.580385 w_k1_IBM_s007
                                 + 0.63077 w_k1_INTC_s007
                                 + 0.682091 w_k1_KO_s007
                                 + 0.791436 w_k1_MAT_s007
                                 + 0.758262 w_k1_MSFT_s007
                                 + 0.859302 w_k1_PG_s007 >= 1
 z_k1_s0100_constraint:          var_1 + z_k1_s0100 + 1.04924 w_k1_ADM_s010
                                 + 1.1527 w_k1_AXP_s010
                                 + 0.950637 w_k1_DIS_s010
                                 + 1.00874 w_k1_FDX_s010
                                 + 0.935572 w_k1_IBM_s010
                                 + 1.17645 w_k1_INTC_s010
                                 + 0.962005 w_k1_KO_s010
                                 + 1.35646 w_k1_MAT_s010
                                 + 0.990753 w_k1_MSFT_s010
                                 + 1.02568 w_k1_PG_s010 >= 1
 z_k1_s0101_constraint:          var_1 + z_k1_s0101 + 1.08777 w_k1_ADM_s010
                                 + 1.11041 w_k1_AXP_s010
                                 + 1.16695 w_k1_DIS_s010 + 1.0661 w_k1_FDX_s010
                                 + 1.45809 w_k1_IBM_s010
                                 + 1.25827 w_k1_INTC_s010
                                 + 1.24974 w_k1_KO_s010 + 1.40599 w_k1_MAT_s010
                                 + 1.11258 w_k1_MSFT_s010
                                 + 1.22285 w_k1_PG_s010 >= 1
 z_k1_s0102_constraint:          var_1 + z_k1_s0102 + 1.02201 w_k1_ADM_s010
                                 + 0.841043 w_k1_AXP_s010
                                 + 1.02419 w_k1_DIS_s010
                                 + 1.03891 w_k1_FDX_s010
                                 + 0.813746 w_k1_IBM_s010
                                 + 1.07931 w_k1_INTC_s010
                                 + 0.933915 w_k1_KO_s010
                                 + 0.893678 w_k1_MAT_s010
                                 + 1.02159 w_k1_MSFT_s010
                                 + 1.02929 w_k1_PG_s010 >= 1
 z_k1_s0103_constraint:          var_1 + z_k1_s0103 + 1.25687 w_k1_ADM_s010
                                 + 1.73401 w_k1_AXP_s010
                                 + 1.82508 w_k1_DIS_s010
                                 + 1.43833 w_k1_FDX_s010
                                 + 1.35063 w_k1_IBM_s010
                                 + 1.78434 w_k1_INTC_s010
                                 + 1.57573 w_k1_KO_s010 + 1.56119 w_k1_MAT_s010
                                 + 1.58601 w_k1_MSFT_s010
                                 + 1.38641 w_k1_PG_s010 >= 1
 z_k1_s0104_constraint:          var_1 + z_k1_s0104 + 1.0742 w_k1_ADM_s010
                                 + 1.18747 w_k1_AXP_s010
                                 + 1.48056 w_k1_DIS_s010
                                 + 1.26955 w_k1_FDX_s010
                                 + 0.832398 w_k1_IBM_s010
                                 + 1.30377 w_k1_INTC_s010
                                 + 1.29734 w_k1_KO_s010 + 1.07225 w_k1_MAT_s010
                                 + 1.25309 w_k1_MSFT_s010
                                 + 1.25701 w_k1_PG_s010 >= 1
 z_k1_s0105_constraint:          var_1 + z_k1_s0105 + 0.954391 w_k1_ADM_s010
                                 + 0.887558 w_k1_AXP_s010
                                 + 1.04421 w_k1_DIS_s010
                                 + 0.845139 w_k1_FDX_s010
                                 + 1.6172 w_k1_IBM_s010
                                 + 1.03836 w_k1_INTC_s010
                                 + 1.32531 w_k1_KO_s010 + 1.77231 w_k1_MAT_s010
                                 + 1.01357 w_k1_MSFT_s010
                                 + 1.30862 w_k1_PG_s010 >= 1
 z_k1_s0106_constraint:          var_1 + z_k1_s0106 + 0.781774 w_k1_ADM_s010
                                 + 0.862746 w_k1_AXP_s010
                                 + 0.874812 w_k1_DIS_s010
                                 + 0.859595 w_k1_FDX_s010
                                 + 1.16345 w_k1_IBM_s010
                                 + 1.04968 w_k1_INTC_s010
                                 + 0.963061 w_k1_KO_s010
                                 + 1.04012 w_k1_MAT_s010
                                 + 0.968841 w_k1_MSFT_s010
                                 + 1.05422 w_k1_PG_s010 >= 1
 z_k1_s0107_constraint:          var_1 + z_k1_s0107 + 0.886257 w_k1_ADM_s010
                                 + 0.674196 w_k1_AXP_s010
                                 + 0.631563 w_k1_DIS_s010
                                 + 0.655245 w_k1_FDX_s010
                                 + 0.948022 w_k1_IBM_s010
                                 + 0.783294 w_k1_INTC_s010
                                 + 0.978128 w_k1_KO_s010
                                 + 0.977682 w_k1_MAT_s010
                                 + 0.718356 w_k1_MSFT_s010
                                 + 0.856762 w_k1_PG_s010 >= 1
 z_k1_s0110_constraint:          var_1 + z_k1_s0110 + 1.18933 w_k1_ADM_s011
                                 + 0.887027 w_k1_AXP_s011
                                 + 0.942847 w_k1_DIS_s011
                                 + 1.06862 w_k1_FDX_s011 + 0.8322 w_k1_IBM_s011
                                 + 1.16446 w_k1_INTC_s011
                                 + 0.953519 w_k1_KO_s011
                                 + 0.643437 w_k1_MAT_s011
                                 + 0.96668 w_k1_MSFT_s011
                                 + 0.925711 w_k1_PG_s011 >= 1
 z_k1_s0111_constraint:          var_1 + z_k1_s0111 + 1.17727 w_k1_ADM_s011
                                 + 1.2512 w_k1_AXP_s011 + 1.31853 w_k1_DIS_s011
                                 + 1.23993 w_k1_FDX_s011
                                 + 0.996856 w_k1_IBM_s011
                                 + 1.19424 w_k1_INTC_s011
                                 + 1.04259 w_k1_KO_s011 + 1.00854 w_k1_MAT_s011
                                 + 1.15443 w_k1_MSFT_s011
                                 + 0.995863 w_k1_PG_s011 >= 1
 z_k1_s0112_constraint:          var_1 + z_k1_s0112 + 1.37373 w_k1_ADM_s011
                                 + 1.39348 w_k1_AXP_s011
                                 + 1.55968 w_k1_DIS_s011
                                 + 1.44501 w_k1_FDX_s011
                                 + 1.13035 w_k1_IBM_s011
                                 + 1.21399 w_k1_INTC_s011
                                 + 1.31038 w_k1_KO_s011 + 1.59088 w_k1_MAT_s011
                                 + 1.36415 w_k1_MSFT_s011
                                 + 1.19488 w_k1_PG_s011 >= 1
 z_k1_s0113_constraint:          var_1 + z_k1_s0113 + 0.766491 w_k1_ADM_s011
                                 + 0.904657 w_k1_AXP_s011
                                 + 0.685698 w_k1_DIS_s011
                                 + 0.877997 w_k1_FDX_s011
                                 + 1.0161 w_k1_IBM_s011
                                 + 1.01664 w_k1_INTC_s011
                                 + 0.827242 w_k1_KO_s011
                                 + 0.918322 w_k1_MAT_s011
                                 + 0.804845 w_k1_MSFT_s011
                                 + 0.930294 w_k1_PG_s011 >= 1
 z_k1_s0114_constraint:          var_1 + z_k1_s0114 + 0.722893 w_k1_ADM_s011
                                 + 1.10452 w_k1_AXP_s011
                                 + 0.791442 w_k1_DIS_s011
                                 + 0.863898 w_k1_FDX_s011
                                 + 1.59491 w_k1_IBM_s011
                                 + 0.916858 w_k1_INTC_s011
                                 + 0.984801 w_k1_KO_s011
                                 + 1.37658 w_k1_MAT_s011
                                 + 0.833924 w_k1_MSFT_s011
                                 + 0.952135 w_k1_PG_s011 >= 1
 z_k1_s0115_constraint:          var_1 + z_k1_s0115 + 1.26314 w_k1_ADM_s011
                                 + 1.12279 w_k1_AXP_s011
                                 + 1.19468 w_k1_DIS_s011
                                 + 1.11426 w_k1_FDX_s011
                                 + 1.35893 w_k1_IBM_s011
                                 + 1.1686 w_k1_INTC_s011 + 1.23892 w_k1_KO_s011
                                 + 1.26724 w_k1_MAT_s011
                                 + 1.22622 w_k1_MSFT_s011
                                 + 1.11868 w_k1_PG_s011 >= 1
 z_k1_s0116_constraint:          var_1 + z_k1_s0116 + 0.980318 w_k1_ADM_s011
                                 + 1.36985 w_k1_AXP_s011
                                 + 1.02223 w_k1_DIS_s011
                                 + 1.09413 w_k1_FDX_s011
                                 + 0.944925 w_k1_IBM_s011
                                 + 1.20184 w_k1_INTC_s011
                                 + 0.996097 w_k1_KO_s011
                                 + 1.15575 w_k1_MAT_s011
                                 + 0.982714 w_k1_MSFT_s011
                                 + 1.04956 w_k1_PG_s011 >= 1
 z_k1_s0117_constraint:          var_1 + z_k1_s0117 + 0.67075 w_k1_ADM_s011
                                 + 0.771467 w_k1_AXP_s011
                                 + 0.632879 w_k1_DIS_s011
                                 + 0.873256 w_k1_FDX_s011
                                 + 0.708086 w_k1_IBM_s011
                                 + 0.938289 w_k1_INTC_s011
                                 + 0.630748 w_k1_KO_s011
                                 + 0.515022 w_k1_MAT_s011
                                 + 0.777225 w_k1_MSFT_s011
                                 + 0.783062 w_k1_PG_s011 >= 1
 z_k1_s0120_constraint:          var_1 + z_k1_s0120 + 0.813994 w_k1_ADM_s012
                                 + 0.753187 w_k1_AXP_s012
                                 + 0.675673 w_k1_DIS_s012
                                 + 0.736365 w_k1_FDX_s012
                                 + 0.748263 w_k1_IBM_s012
                                 + 0.769927 w_k1_INTC_s012
                                 + 0.765899 w_k1_KO_s012
                                 + 0.652464 w_k1_MAT_s012
                                 + 0.722484 w_k1_MSFT_s012
                                 + 0.821422 w_k1_PG_s012 >= 1
 z_k1_s0121_constraint:          var_1 + z_k1_s0121 + 1.30383 w_k1_ADM_s012
                                 + 1.46574 w_k1_AXP_s012
                                 + 1.37161 w_k1_DIS_s012
                                 + 1.07987 w_k1_FDX_s012
                                 + 1.61603 w_k1_IBM_s012
                                 + 1.34098 w_k1_INTC_s012
                                 + 1.43801 w_k1_KO_s012 + 1.5071 w_k1_MAT_s012
                                 + 1.08641 w_k1_MSFT_s012
                                 + 1.24509 w_k1_PG_s012 >= 1
 z_k1_s0122_constraint:          var_1 + z_k1_s0122 + 1.46517 w_k1_ADM_s012
                                 + 1.66163 w_k1_AXP_s012
                                 + 1.62581 w_k1_DIS_s012
                                 + 1.30555 w_k1_FDX_s012
                                 + 1.00229 w_k1_IBM_s012
                                 + 1.24045 w_k1_INTC_s012
                                 + 1.21422 w_k1_KO_s012 + 1.25259 w_k1_MAT_s012
                                 + 1.22143 w_k1_MSFT_s012
                                 + 1.34531 w_k1_PG_s012 >= 1
 z_k1_s0123_constraint:          var_1 + z_k1_s0123 + 1.36434 w_k1_ADM_s012
                                 + 1.12145 w_k1_AXP_s012
                                 + 1.14003 w_k1_DIS_s012
                                 + 0.961673 w_k1_FDX_s012
                                 + 1.38434 w_k1_IBM_s012
                                 + 0.935394 w_k1_INTC_s012
                                 + 1.1133 w_k1_KO_s012 + 0.981721 w_k1_MAT_s012
                                 + 0.870492 w_k1_MSFT_s012
                                 + 1.19655 w_k1_PG_s012 >= 1
 z_k1_s0124_constraint:          var_1 + z_k1_s0124 + 0.89134 w_k1_ADM_s012
                                 + 1.01336 w_k1_AXP_s012
                                 + 0.965204 w_k1_DIS_s012
                                 + 0.886824 w_k1_FDX_s012
                                 + 1.13326 w_k1_IBM_s012
                                 + 1.02664 w_k1_INTC_s012
                                 + 1.04084 w_k1_KO_s012
                                 + 0.967839 w_k1_MAT_s012
                                 + 0.835692 w_k1_MSFT_s012
                                 + 1.09344 w_k1_PG_s012 >= 1
 z_k1_s0125_constraint:          var_1 + z_k1_s0125 + 1.10573 w_k1_ADM_s012
                                 + 1.07534 w_k1_AXP_s012
                                 + 0.908082 w_k1_DIS_s012
                                 + 1.04096 w_k1_FDX_s012
                                 + 0.751709 w_k1_IBM_s012
                                 + 0.915398 w_k1_INTC_s012
                                 + 0.911083 w_k1_KO_s012
                                 + 0.841986 w_k1_MAT_s012
                                 + 0.925146 w_k1_MSFT_s012
                                 + 1.13641 w_k1_PG_s012 >= 1
 z_k1_s0126_constraint:          var_1 + z_k1_s0126 + 1.14046 w_k1_ADM_s012
                                 + 1.19277 w_k1_AXP_s012
                                 + 1.24083 w_k1_DIS_s012
                                 + 1.29173 w_k1_FDX_s012
                                 + 0.672656 w_k1_IBM_s012
                                 + 1.01874 w_k1_INTC_s012
                                 + 1.04665 w_k1_KO_s012
                                 + 0.911342 w_k1_MAT_s012
                                 + 1.06277 w_k1_MSFT_s012
                                 + 1.10819 w_k1_PG_s012 >= 1
 z_k1_s0127_constraint:          var_1 + z_k1_s0127 + 1.14365 w_k1_ADM_s012
                                 + 0.770561 w_k1_AXP_s012
                                 + 1.38517 w_k1_DIS_s012
                                 + 1.06433 w_k1_FDX_s012
                                 + 0.800477 w_k1_IBM_s012
                                 + 0.978537 w_k1_INTC_s012
                                 + 1.01359 w_k1_KO_s012
                                 + 0.483781 w_k1_MAT_s012
                                 + 1.00963 w_k1_MSFT_s012
                                 + 1.04788 w_k1_PG_s012 >= 1
 z_k1_s0130_constraint:          var_1 + z_k1_s0130 + 1.0412 w_k1_ADM_s013
                                 + 0.808823 w_k1_AXP_s013
                                 + 1.14261 w_k1_DIS_s013
                                 + 1.00821 w_k1_FDX_s013
                                 + 0.962711 w_k1_IBM_s013
                                 + 1.03954 w_k1_INTC_s013
                                 + 1.09659 w_k1_KO_s013
                                 + 0.675355 w_k1_MAT_s013
                                 + 1.07622 w_k1_MSFT_s013
                                 + 1.05035 w_k1_PG_s013 >= 1
 z_k1_s0131_constraint:          var_1 + z_k1_s0131 + 0.956314 w_k1_ADM_s013
                                 + 0.893063 w_k1_AXP_s013
                                 + 1.05653 w_k1_DIS_s013
                                 + 1.15837 w_k1_FDX_s013
                                 + 0.649273 w_k1_IBM_s013
                                 + 1.33049 w_k1_INTC_s013
                                 + 0.993581 w_k1_KO_s013
                                 + 0.671402 w_k1_MAT_s013
                                 + 1.17809 w_k1_MSFT_s013
                                 + 0.959218 w_k1_PG_s013 >= 1
 z_k1_s0132_constraint:          var_1 + z_k1_s0132 + 0.941793 w_k1_ADM_s013
                                 + 0.925001 w_k1_AXP_s013
                                 + 0.761065 w_k1_DIS_s013
                                 + 0.955285 w_k1_FDX_s013
                                 + 0.739332 w_k1_IBM_s013
                                 + 1.07232 w_k1_INTC_s013
                                 + 0.812025 w_k1_KO_s013
                                 + 0.843858 w_k1_MAT_s013
                                 + 0.849805 w_k1_MSFT_s013
                                 + 0.943051 w_k1_PG_s013 >= 1
 z_k1_s0133_constraint:          var_1 + z_k1_s0133 + 1.00317 w_k1_ADM_s013
                                 + 1.11426 w_k1_AXP_s013
                                 + 0.968423 w_k1_DIS_s013
                                 + 0.950324 w_k1_FDX_s013
                                 + 1.24481 w_k1_IBM_s013
                                 + 0.973753 w_k1_INTC_s013
                                 + 1.11629 w_k1_KO_s013 + 1.50396 w_k1_MAT_s013
                                 + 0.856519 w_k1_MSFT_s013
                                 + 1.08221 w_k1_PG_s013 >= 1
 z_k1_s0134_constraint:          var_1 + z_k1_s0134 + 1.32022 w_k1_ADM_s013
                                 + 1.32994 w_k1_AXP_s013
                                 + 1.24876 w_k1_DIS_s013
                                 + 1.18118 w_k1_FDX_s013
                                 + 0.921292 w_k1_IBM_s013
                                 + 1.04743 w_k1_INTC_s013
                                 + 1.13186 w_k1_KO_s013 + 1.55579 w_k1_MAT_s013
                                 + 1.01551 w_k1_MSFT_s013
                                 + 1.11797 w_k1_PG_s013 >= 1
 z_k1_s0135_constraint:          var_1 + z_k1_s0135 + 1.26381 w_k1_ADM_s013
                                 + 1.32297 w_k1_AXP_s013
                                 + 1.53516 w_k1_DIS_s013
                                 + 1.29842 w_k1_FDX_s013
                                 + 1.34149 w_k1_IBM_s013
                                 + 1.44473 w_k1_INTC_s013
                                 + 1.38969 w_k1_KO_s013
                                 + 0.937096 w_k1_MAT_s013
                                 + 1.1896 w_k1_MSFT_s013 + 1.18929 w_k1_PG_s013
                                 >= 1
 z_k1_s0136_constraint:          var_1 + z_k1_s0136 + 0.882937 w_k1_ADM_s013
                                 + 1.13483 w_k1_AXP_s013
                                 + 0.807853 w_k1_DIS_s013
                                 + 0.789513 w_k1_FDX_s013
                                 + 1.3463 w_k1_IBM_s013
                                 + 0.790833 w_k1_INTC_s013
                                 + 1.17669 w_k1_KO_s013 + 1.82205 w_k1_MAT_s013
                                 + 0.803953 w_k1_MSFT_s013
                                 + 0.998767 w_k1_PG_s013 >= 1
 z_k1_s0137_constraint:          var_1 + z_k1_s0137 + 0.742219 w_k1_ADM_s013
                                 + 0.632863 w_k1_AXP_s013
                                 + 0.72755 w_k1_DIS_s013
                                 + 0.748859 w_k1_FDX_s013
                                 + 0.794768 w_k1_IBM_s013
                                 + 0.947739 w_k1_INTC_s013
                                 + 0.88857 w_k1_KO_s013
                                 + 0.706318 w_k1_MAT_s013
                                 + 0.800862 w_k1_MSFT_s013
                                 + 0.823592 w_k1_PG_s013 >= 1
 z_k1_s0140_constraint:          var_1 + z_k1_s0140 + 0.859966 w_k1_ADM_s014
                                 + 0.664146 w_k1_AXP_s014
                                 + 0.852531 w_k1_DIS_s014
                                 + 0.873388 w_k1_FDX_s014
                                 + 1.10883 w_k1_IBM_s014
                                 + 0.772725 w_k1_INTC_s014
                                 + 0.938166 w_k1_KO_s014
                                 + 1.09359 w_k1_MAT_s014
                                 + 0.885384 w_k1_MSFT_s014
                                 + 0.970309 w_k1_PG_s014 >= 1
 z_k1_s0141_constraint:          var_1 + z_k1_s0141 + 1.01593 w_k1_ADM_s014
                                 + 1.48887 w_k1_AXP_s014
                                 + 1.66068 w_k1_DIS_s014
                                 + 1.40937 w_k1_FDX_s014
                                 + 1.15978 w_k1_IBM_s014
                                 + 1.02307 w_k1_INTC_s014
                                 + 1.38525 w_k1_KO_s014 + 1.7273 w_k1_MAT_s014
                                 + 1.34227 w_k1_MSFT_s014
                                 + 1.21116 w_k1_PG_s014 >= 1
 z_k1_s0142_constraint:          var_1 + z_k1_s0142 + 0.876913 w_k1_ADM_s014
                                 + 1.1017 w_k1_AXP_s014 + 1.05747 w_k1_DIS_s014
                                 + 1.07221 w_k1_FDX_s014
                                 + 1.15683 w_k1_IBM_s014
                                 + 0.940312 w_k1_INTC_s014
                                 + 1.14586 w_k1_KO_s014 + 1.30067 w_k1_MAT_s014
                                 + 0.938465 w_k1_MSFT_s014
                                 + 0.945815 w_k1_PG_s014 >= 1
 z_k1_s0143_constraint:          var_1 + z_k1_s0143 + 0.827717 w_k1_ADM_s014
                                 + 1.00398 w_k1_AXP_s014
                                 + 0.888371 w_k1_DIS_s014
                                 + 0.912228 w_k1_FDX_s014
                                 + 1.00095 w_k1_IBM_s014
                                 + 1.07696 w_k1_INTC_s014
                                 + 0.978268 w_k1_KO_s014
                                 + 0.999067 w_k1_MAT_s014
                                 + 0.957555 w_k1_MSFT_s014
                                 + 0.969582 w_k1_PG_s014 >= 1
 z_k1_s0144_constraint:          var_1 + z_k1_s0144 + 0.774132 w_k1_ADM_s014
                                 + 0.958879 w_k1_AXP_s014
                                 + 1.27083 w_k1_DIS_s014
                                 + 1.00172 w_k1_FDX_s014 + 1.6606 w_k1_IBM_s014
                                 + 1.03458 w_k1_INTC_s014
                                 + 1.44268 w_k1_KO_s014 + 1.54468 w_k1_MAT_s014
                                 + 1.12367 w_k1_MSFT_s014
                                 + 1.23819 w_k1_PG_s014 >= 1
 z_k1_s0145_constraint:          var_1 + z_k1_s0145 + 0.905831 w_k1_ADM_s014
                                 + 0.930001 w_k1_AXP_s014
                                 + 0.85024 w_k1_DIS_s014
                                 + 0.961716 w_k1_FDX_s014
                                 + 0.757533 w_k1_IBM_s014
                                 + 1.01123 w_k1_INTC_s014
                                 + 0.804848 w_k1_KO_s014 + 0.6487 w_k1_MAT_s014
                                 + 0.937611 w_k1_MSFT_s014
                                 + 0.978232 w_k1_PG_s014 >= 1
 z_k1_s0146_constraint:          var_1 + z_k1_s0146 + 1.14901 w_k1_ADM_s014
                                 + 1.41734 w_k1_AXP_s014
                                 + 1.24237 w_k1_DIS_s014
                                 + 1.25011 w_k1_FDX_s014
                                 + 1.15319 w_k1_IBM_s014
                                 + 1.1496 w_k1_INTC_s014 + 1.09021 w_k1_KO_s014
                                 + 1.12472 w_k1_MAT_s014
                                 + 1.07475 w_k1_MSFT_s014
                                 + 1.07518 w_k1_PG_s014 >= 1
 z_k1_s0147_constraint:          var_1 + z_k1_s0147 + 0.534305 w_k1_ADM_s014
                                 + 0.593316 w_k1_AXP_s014
                                 + 0.674803 w_k1_DIS_s014
                                 + 0.745526 w_k1_FDX_s014
                                 + 0.719802 w_k1_IBM_s014
                                 + 0.789975 w_k1_INTC_s014
                                 + 0.714436 w_k1_KO_s014
                                 + 0.721796 w_k1_MAT_s014
                                 + 0.831337 w_k1_MSFT_s014
                                 + 0.801382 w_k1_PG_s014 >= 1
 z_k1_s0150_constraint:          var_1 + z_k1_s0150 + 0.909945 w_k1_ADM_s015
                                 + 1.12257 w_k1_AXP_s015
                                 + 0.96688 w_k1_DIS_s015
                                 + 0.941825 w_k1_FDX_s015
                                 + 1.42203 w_k1_IBM_s015
                                 + 1.10222 w_k1_INTC_s015
                                 + 1.27288 w_k1_KO_s015 + 1.80214 w_k1_MAT_s015
                                 + 0.919565 w_k1_MSFT_s015
                                 + 1.09848 w_k1_PG_s015 >= 1
 z_k1_s0151_constraint:          var_1 + z_k1_s0151 + 1.14453 w_k1_ADM_s015
                                 + 1.25925 w_k1_AXP_s015
                                 + 1.54631 w_k1_DIS_s015 + 1.3272 w_k1_FDX_s015
                                 + 1.68598 w_k1_IBM_s015
                                 + 1.26532 w_k1_INTC_s015
                                 + 1.37297 w_k1_KO_s015 + 1.74505 w_k1_MAT_s015
                                 + 1.21574 w_k1_MSFT_s015
                                 + 1.24409 w_k1_PG_s015 >= 1
 z_k1_s0152_constraint:          var_1 + z_k1_s0152 + 1.14126 w_k1_ADM_s015
                                 + 1.16714 w_k1_AXP_s015
                                 + 1.56199 w_k1_DIS_s015
                                 + 1.25522 w_k1_FDX_s015
                                 + 0.835163 w_k1_IBM_s015
                                 + 1.46518 w_k1_INTC_s015
                                 + 1.19247 w_k1_KO_s015
                                 + 0.815575 w_k1_MAT_s015
                                 + 1.30035 w_k1_MSFT_s015
                                 + 1.11378 w_k1_PG_s015 >= 1
 z_k1_s0153_constraint:          var_1 + z_k1_s0153 + 0.717509 w_k1_ADM_s015
                                 + 0.730831 w_k1_AXP_s015
                                 + 0.626835 w_k1_DIS_s015
                                 + 0.768827 w_k1_FDX_s015
                                 + 0.821482 w_k1_IBM_s015
                                 + 0.891687 w_k1_INTC_s015
                                 + 0.721534 w_k1_KO_s015
                                 + 0.926865 w_k1_MAT_s015
                                 + 0.758054 w_k1_MSFT_s015
                                 + 0.856746 w_k1_PG_s015 >= 1
 z_k1_s0154_constraint:          var_1 + z_k1_s0154 + 1.04677 w_k1_ADM_s015
                                 + 1.05633 w_k1_AXP_s015
                                 + 0.810979 w_k1_DIS_s015
                                 + 0.9317 w_k1_FDX_s015
                                 + 0.723716 w_k1_IBM_s015
                                 + 1.10446 w_k1_INTC_s015
                                 + 0.812881 w_k1_KO_s015
                                 + 0.931911 w_k1_MAT_s015
                                 + 0.895148 w_k1_MSFT_s015
                                 + 0.952749 w_k1_PG_s015 >= 1
 z_k1_s0155_constraint:          var_1 + z_k1_s0155 + 0.894433 w_k1_ADM_s015
                                 + 1.07541 w_k1_AXP_s015
                                 + 0.952048 w_k1_DIS_s015
                                 + 1.03352 w_k1_FDX_s015
                                 + 0.990082 w_k1_IBM_s015
                                 + 1.15284 w_k1_INTC_s015
                                 + 0.942237 w_k1_KO_s015
                                 + 0.897997 w_k1_MAT_s015
                                 + 0.973336 w_k1_MSFT_s015
                                 + 0.98296 w_k1_PG_s015 >= 1
 z_k1_s0156_constraint:          var_1 + z_k1_s0156 + 1.02376 w_k1_ADM_s015
                                 + 1.14147 w_k1_AXP_s015
                                 + 1.42438 w_k1_DIS_s015 + 1.2183 w_k1_FDX_s015
                                 + 1.18118 w_k1_IBM_s015
                                 + 1.18059 w_k1_INTC_s015
                                 + 1.21907 w_k1_KO_s015 + 1.30342 w_k1_MAT_s015
                                 + 1.29151 w_k1_MSFT_s015
                                 + 1.20673 w_k1_PG_s015 >= 1
 z_k1_s0157_constraint:          var_1 + z_k1_s0157 + 0.9087 w_k1_ADM_s015
                                 + 0.840404 w_k1_AXP_s015
                                 + 0.959124 w_k1_DIS_s015
                                 + 1.00259 w_k1_FDX_s015
                                 + 0.581044 w_k1_IBM_s015
                                 + 1.27953 w_k1_INTC_s015
                                 + 0.785519 w_k1_KO_s015
                                 + 0.520033 w_k1_MAT_s015
                                 + 1.00308 w_k1_MSFT_s015
                                 + 0.968593 w_k1_PG_s015 >= 1
 z_k1_s0160_constraint:          var_1 + z_k1_s0160 + 0.828546 w_k1_ADM_s016
                                 + 0.688052 w_k1_AXP_s016
                                 + 0.613164 w_k1_DIS_s016
                                 + 0.671675 w_k1_FDX_s016
                                 + 1.00646 w_k1_IBM_s016
                                 + 0.693905 w_k1_INTC_s016
                                 + 0.803713 w_k1_KO_s016
                                 + 0.963304 w_k1_MAT_s016
                                 + 0.733454 w_k1_MSFT_s016
                                 + 0.868901 w_k1_PG_s016 >= 1
 z_k1_s0161_constraint:          var_1 + z_k1_s0161 + 0.887427 w_k1_ADM_s016
                                 + 0.847978 w_k1_AXP_s016
                                 + 0.811888 w_k1_DIS_s016
                                 + 0.869588 w_k1_FDX_s016
                                 + 0.857899 w_k1_IBM_s016
                                 + 0.897794 w_k1_INTC_s016
                                 + 0.836689 w_k1_KO_s016
                                 + 0.869508 w_k1_MAT_s016
                                 + 0.796167 w_k1_MSFT_s016
                                 + 0.925421 w_k1_PG_s016 >= 1
 z_k1_s0162_constraint:          var_1 + z_k1_s0162 + 0.952311 w_k1_ADM_s016
                                 + 1.06791 w_k1_AXP_s016
                                 + 1.06973 w_k1_DIS_s016
                                 + 1.03457 w_k1_FDX_s016
                                 + 1.15106 w_k1_IBM_s016
                                 + 0.876673 w_k1_INTC_s016
                                 + 1.11162 w_k1_KO_s016 + 1.27119 w_k1_MAT_s016
                                 + 1.0107 w_k1_MSFT_s016 + 1.11222 w_k1_PG_s016
                                 >= 1
 z_k1_s0163_constraint:          var_1 + z_k1_s0163 + 1.33916 w_k1_ADM_s016
                                 + 1.64848 w_k1_AXP_s016
                                 + 1.52318 w_k1_DIS_s016
                                 + 1.40668 w_k1_FDX_s016
                                 + 1.32616 w_k1_IBM_s016
                                 + 1.50033 w_k1_INTC_s016
                                 + 1.35088 w_k1_KO_s016 + 1.3532 w_k1_MAT_s016
                                 + 1.30821 w_k1_MSFT_s016
                                 + 1.11278 w_k1_PG_s016 >= 1
 z_k1_s0164_constraint:          var_1 + z_k1_s0164 + 1.06046 w_k1_ADM_s016
                                 + 0.885309 w_k1_AXP_s016
                                 + 1.40583 w_k1_DIS_s016
                                 + 1.05538 w_k1_FDX_s016
                                 + 1.23845 w_k1_IBM_s016
                                 + 1.06443 w_k1_INTC_s016
                                 + 1.11346 w_k1_KO_s016
                                 + 0.682774 w_k1_MAT_s016
                                 + 1.07406 w_k1_MSFT_s016
                                 + 0.905905 w_k1_PG_s016 >= 1
 z_k1_s0165_constraint:          var_1 + z_k1_s0165 + 0.870358 w_k1_ADM_s016
                                 + 0.59521 w_k1_AXP_s016
                                 + 0.659136 w_k1_DIS_s016
                                 + 0.772547 w_k1_FDX_s016
                                 + 0.68585 w_k1_IBM_s016
                                 + 0.833129 w_k1_INTC_s016
                                 + 0.658074 w_k1_KO_s016
                                 + 0.479181 w_k1_MAT_s016
                                 + 0.745815 w_k1_MSFT_s016
                                 + 0.811995 w_k1_PG_s016 >= 1
 z_k1_s0166_constraint:          var_1 + z_k1_s0166 + 1.31743 w_k1_ADM_s016
                                 + 1.2266 w_k1_AXP_s016
                                 + 0.910343 w_k1_DIS_s016
                                 + 1.16192 w_k1_FDX_s016
                                 + 0.795309 w_k1_IBM_s016
                                 + 1.05273 w_k1_INTC_s016
                                 + 0.862081 w_k1_KO_s016 + 1.0916 w_k1_MAT_s016
                                 + 0.917966 w_k1_MSFT_s016
                                 + 0.822817 w_k1_PG_s016 >= 1
 z_k1_s0167_constraint:          var_1 + z_k1_s0167 + 1.0833 w_k1_ADM_s016
                                 + 1.46218 w_k1_AXP_s016
                                 + 1.05215 w_k1_DIS_s016
                                 + 0.970809 w_k1_FDX_s016
                                 + 1.29104 w_k1_IBM_s016
                                 + 0.850208 w_k1_INTC_s016
                                 + 1.0693 w_k1_KO_s016 + 1.7385 w_k1_MAT_s016
                                 + 0.788061 w_k1_MSFT_s016
                                 + 0.959025 w_k1_PG_s016 >= 1
 z_k1_s0170_constraint:          var_1 + z_k1_s0170 + 1.03489 w_k1_ADM_s017
                                 + 0.954916 w_k1_AXP_s017
                                 + 1.02503 w_k1_DIS_s017 + 1.0397 w_k1_FDX_s017
                                 + 1.11644 w_k1_IBM_s017
                                 + 1.02404 w_k1_INTC_s017
                                 + 1.15076 w_k1_KO_s017 + 1.05387 w_k1_MAT_s017
                                 + 1.11472 w_k1_MSFT_s017
                                 + 1.09785 w_k1_PG_s017 >= 1
 z_k1_s0171_constraint:          var_1 + z_k1_s0171 + 1.19718 w_k1_ADM_s017
                                 + 1.31093 w_k1_AXP_s017
                                 + 1.53619 w_k1_DIS_s017 + 1.3085 w_k1_FDX_s017
                                 + 1.26039 w_k1_IBM_s017
                                 + 1.35536 w_k1_INTC_s017
                                 + 1.57142 w_k1_KO_s017 + 1.2029 w_k1_MAT_s017
                                 + 1.37978 w_k1_MSFT_s017
                                 + 1.24918 w_k1_PG_s017 >= 1
 z_k1_s0172_constraint:          var_1 + z_k1_s0172 + 0.868807 w_k1_ADM_s017
                                 + 0.763907 w_k1_AXP_s017
                                 + 0.786573 w_k1_DIS_s017
                                 + 0.934884 w_k1_FDX_s017
                                 + 0.708356 w_k1_IBM_s017
                                 + 0.866663 w_k1_INTC_s017
                                 + 0.778146 w_k1_KO_s017
                                 + 0.819257 w_k1_MAT_s017
                                 + 0.97889 w_k1_MSFT_s017
                                 + 0.836347 w_k1_PG_s017 >= 1
 z_k1_s0173_constraint:          var_1 + z_k1_s0173 + 0.683049 w_k1_ADM_s017
                                 + 0.718352 w_k1_AXP_s017
                                 + 0.538529 w_k1_DIS_s017
                                 + 0.672765 w_k1_FDX_s017
                                 + 0.728313 w_k1_IBM_s017
                                 + 0.760548 w_k1_INTC_s017
                                 + 0.679523 w_k1_KO_s017 + 0.5609 w_k1_MAT_s017
                                 + 0.7467 w_k1_MSFT_s017
                                 + 0.730503 w_k1_PG_s017 >= 1
 z_k1_s0174_constraint:          var_1 + z_k1_s0174 + 0.954108 w_k1_ADM_s017
                                 + 0.785847 w_k1_AXP_s017
                                 + 0.790285 w_k1_DIS_s017
                                 + 0.820601 w_k1_FDX_s017
                                 + 0.94028 w_k1_IBM_s017
                                 + 0.912933 w_k1_INTC_s017
                                 + 0.936782 w_k1_KO_s017
                                 + 0.73851 w_k1_MAT_s017
                                 + 0.943523 w_k1_MSFT_s017
                                 + 0.932086 w_k1_PG_s017 >= 1
 z_k1_s0175_constraint:          var_1 + z_k1_s0175 + 1.11746 w_k1_ADM_s017
                                 + 0.801339 w_k1_AXP_s017
                                 + 1.10853 w_k1_DIS_s017
                                 + 1.02577 w_k1_FDX_s017
                                 + 0.889981 w_k1_IBM_s017
                                 + 0.966212 w_k1_INTC_s017
                                 + 1.12567 w_k1_KO_s017
                                 + 0.826437 w_k1_MAT_s017
                                 + 1.14866 w_k1_MSFT_s017
                                 + 1.00755 w_k1_PG_s017 >= 1
 z_k1_s0176_constraint:          var_1 + z_k1_s0176 + 1.08313 w_k1_ADM_s017
                                 + 1.01025 w_k1_AXP_s017
                                 + 1.34994 w_k1_DIS_s017
                                 + 1.03095 w_k1_FDX_s017
                                 + 1.27547 w_k1_IBM_s017
                                 + 1.17278 w_k1_INTC_s017
                                 + 1.22865 w_k1_KO_s017 + 1.05793 w_k1_MAT_s017
                                 + 1.19082 w_k1_MSFT_s017
                                 + 1.05644 w_k1_PG_s017 >= 1
 z_k1_s0177_constraint:          var_1 + z_k1_s0177 + 1.0327 w_k1_ADM_s017
                                 + 1.03841 w_k1_AXP_s017
                                 + 1.32615 w_k1_DIS_s017
                                 + 1.03964 w_k1_FDX_s017
                                 + 1.75122 w_k1_IBM_s017
                                 + 1.06252 w_k1_INTC_s017
                                 + 1.58732 w_k1_KO_s017 + 1.57323 w_k1_MAT_s017
                                 + 1.148 w_k1_MSFT_s017 + 1.18093 w_k1_PG_s017
                                 >= 1
 z_k1_s1000_constraint:          var_1 + z_k1_s1000 + 0.482458 w_k1_ADM_s100
                                 + 0.83561 w_k1_AXP_s100
                                 + 0.585556 w_k1_DIS_s100
                                 + 0.716834 w_k1_FDX_s100
                                 + 0.953795 w_k1_IBM_s100
                                 + 1.0253 w_k1_INTC_s100
                                 + 0.854345 w_k1_KO_s100
                                 + 0.96093 w_k1_MAT_s100
                                 + 0.866917 w_k1_MSFT_s100
                                 + 0.864864 w_k1_PG_s100 >= 1
 z_k1_s1001_constraint:          var_1 + z_k1_s1001 + 0.907004 w_k1_ADM_s100
                                 + 0.830539 w_k1_AXP_s100
                                 + 0.766282 w_k1_DIS_s100
                                 + 0.839157 w_k1_FDX_s100
                                 + 0.765825 w_k1_IBM_s100
                                 + 0.801723 w_k1_INTC_s100
                                 + 0.808623 w_k1_KO_s100
                                 + 0.753384 w_k1_MAT_s100
                                 + 0.871283 w_k1_MSFT_s100
                                 + 0.97778 w_k1_PG_s100 >= 1
 z_k1_s1002_constraint:          var_1 + z_k1_s1002 + 1.00844 w_k1_ADM_s100
                                 + 1.15242 w_k1_AXP_s100
                                 + 1.44265 w_k1_DIS_s100
                                 + 1.27838 w_k1_FDX_s100
                                 + 0.911626 w_k1_IBM_s100
                                 + 1.3817 w_k1_INTC_s100 + 1.04123 w_k1_KO_s100
                                 + 1.24139 w_k1_MAT_s100
                                 + 1.33207 w_k1_MSFT_s100
                                 + 1.17468 w_k1_PG_s100 >= 1
 z_k1_s1003_constraint:          var_1 + z_k1_s1003 + 0.911274 w_k1_ADM_s100
                                 + 1.56482 w_k1_AXP_s100
                                 + 1.64823 w_k1_DIS_s100
                                 + 1.32193 w_k1_FDX_s100
                                 + 1.40557 w_k1_IBM_s100
                                 + 1.49223 w_k1_INTC_s100 + 1.4874 w_k1_KO_s100
                                 + 1.53079 w_k1_MAT_s100
                                 + 1.51448 w_k1_MSFT_s100
                                 + 1.32787 w_k1_PG_s100 >= 1
 z_k1_s1004_constraint:          var_1 + z_k1_s1004 + 1.01095 w_k1_ADM_s100
                                 + 1.40654 w_k1_AXP_s100
                                 + 1.24927 w_k1_DIS_s100 + 1.1001 w_k1_FDX_s100
                                 + 1.17401 w_k1_IBM_s100
                                 + 1.11328 w_k1_INTC_s100
                                 + 1.22991 w_k1_KO_s100 + 1.47229 w_k1_MAT_s100
                                 + 1.19396 w_k1_MSFT_s100
                                 + 1.14427 w_k1_PG_s100 >= 1
 z_k1_s1005_constraint:          var_1 + z_k1_s1005 + 0.838137 w_k1_ADM_s100
                                 + 1.0872 w_k1_AXP_s100 + 1.1592 w_k1_DIS_s100
                                 + 0.986162 w_k1_FDX_s100
                                 + 0.938595 w_k1_IBM_s100
                                 + 0.93428 w_k1_INTC_s100
                                 + 1.06955 w_k1_KO_s100 + 1.27485 w_k1_MAT_s100
                                 + 1.13953 w_k1_MSFT_s100
                                 + 1.07667 w_k1_PG_s100 >= 1
 z_k1_s1006_constraint:          var_1 + z_k1_s1006 + 1.04657 w_k1_ADM_s100
                                 + 1.20172 w_k1_AXP_s100
                                 + 1.21248 w_k1_DIS_s100
                                 + 1.14475 w_k1_FDX_s100
                                 + 0.816505 w_k1_IBM_s100
                                 + 1.0884 w_k1_INTC_s100
                                 + 0.984181 w_k1_KO_s100
                                 + 0.989154 w_k1_MAT_s100
                                 + 1.16302 w_k1_MSFT_s100
                                 + 1.12737 w_k1_PG_s100 >= 1
 z_k1_s1007_constraint:          var_1 + z_k1_s1007 + 0.805875 w_k1_ADM_s100
                                 + 1.06527 w_k1_AXP_s100
                                 + 0.88403 w_k1_DIS_s100
                                 + 0.909397 w_k1_FDX_s100
                                 + 1.21243 w_k1_IBM_s100
                                 + 1.08274 w_k1_INTC_s100
                                 + 1.15588 w_k1_KO_s100 + 1.16203 w_k1_MAT_s100
                                 + 0.865688 w_k1_MSFT_s100
                                 + 0.897905 w_k1_PG_s100 >= 1
 z_k1_s1010_constraint:          var_1 + z_k1_s1010 + 0.933625 w_k1_ADM_s101
                                 + 0.776448 w_k1_AXP_s101
                                 + 0.932458 w_k1_DIS_s101
                                 + 0.957963 w_k1_FDX_s101
                                 + 0.809832 w_k1_IBM_s101
                                 + 0.978309 w_k1_INTC_s101
                                 + 0.920951 w_k1_KO_s101
                                 + 0.706255 w_k1_MAT_s101
                                 + 1.00653 w_k1_MSFT_s101
                                 + 0.914893 w_k1_PG_s101 >= 1
 z_k1_s1011_constraint:          var_1 + z_k1_s1011 + 0.840039 w_k1_ADM_s101
                                 + 0.745215 w_k1_AXP_s101
                                 + 0.768571 w_k1_DIS_s101
                                 + 0.815356 w_k1_FDX_s101
                                 + 1.13569 w_k1_IBM_s101
                                 + 0.764943 w_k1_INTC_s101
                                 + 0.954596 w_k1_KO_s101 + 1.42 w_k1_MAT_s101
                                 + 0.844385 w_k1_MSFT_s101
                                 + 0.813613 w_k1_PG_s101 >= 1
 z_k1_s1012_constraint:          var_1 + z_k1_s1012 + 0.82036 w_k1_ADM_s101
                                 + 0.644074 w_k1_AXP_s101
                                 + 0.660196 w_k1_DIS_s101
                                 + 0.7355 w_k1_FDX_s101
                                 + 0.811835 w_k1_IBM_s101
                                 + 0.644326 w_k1_INTC_s101
                                 + 0.716774 w_k1_KO_s101
                                 + 0.882228 w_k1_MAT_s101
                                 + 0.760134 w_k1_MSFT_s101
                                 + 0.81176 w_k1_PG_s101 >= 1
 z_k1_s1013_constraint:          var_1 + z_k1_s1013 + 1.38243 w_k1_ADM_s101
                                 + 1.52078 w_k1_AXP_s101
                                 + 1.52115 w_k1_DIS_s101
                                 + 1.41607 w_k1_FDX_s101
                                 + 1.30828 w_k1_IBM_s101
                                 + 1.31241 w_k1_INTC_s101
                                 + 1.26581 w_k1_KO_s101 + 1.2542 w_k1_MAT_s101
                                 + 1.46738 w_k1_MSFT_s101
                                 + 1.32289 w_k1_PG_s101 >= 1
 z_k1_s1014_constraint:          var_1 + z_k1_s1014 + 1.00011 w_k1_ADM_s101
                                 + 0.944127 w_k1_AXP_s101
                                 + 0.919467 w_k1_DIS_s101
                                 + 0.932115 w_k1_FDX_s101
                                 + 1.56758 w_k1_IBM_s101
                                 + 1.08226 w_k1_INTC_s101
                                 + 1.22542 w_k1_KO_s101 + 1.15347 w_k1_MAT_s101
                                 + 1.01997 w_k1_MSFT_s101
                                 + 1.03585 w_k1_PG_s101 >= 1
 z_k1_s1015_constraint:          var_1 + z_k1_s1015 + 1.21032 w_k1_ADM_s101
                                 + 1.06788 w_k1_AXP_s101
                                 + 0.893636 w_k1_DIS_s101
                                 + 1.10257 w_k1_FDX_s101
                                 + 0.866836 w_k1_IBM_s101
                                 + 1.14372 w_k1_INTC_s101
                                 + 1.01503 w_k1_KO_s101 + 1.2799 w_k1_MAT_s101
                                 + 1.26008 w_k1_MSFT_s101
                                 + 1.01333 w_k1_PG_s101 >= 1
 z_k1_s1016_constraint:          var_1 + z_k1_s1016 + 1.19627 w_k1_ADM_s101
                                 + 1.12492 w_k1_AXP_s101
                                 + 1.28321 w_k1_DIS_s101
                                 + 1.09775 w_k1_FDX_s101
                                 + 1.09501 w_k1_IBM_s101
                                 + 0.975118 w_k1_INTC_s101
                                 + 1.20205 w_k1_KO_s101 + 1.14751 w_k1_MAT_s101
                                 + 1.19731 w_k1_MSFT_s101
                                 + 0.988251 w_k1_PG_s101 >= 1
 z_k1_s1017_constraint:          var_1 + z_k1_s1017 + 0.923122 w_k1_ADM_s101
                                 + 0.863364 w_k1_AXP_s101
                                 + 1.09836 w_k1_DIS_s101
                                 + 0.987346 w_k1_FDX_s101
                                 + 1.12452 w_k1_IBM_s101
                                 + 0.897902 w_k1_INTC_s101
                                 + 1.15264 w_k1_KO_s101 + 1.1102 w_k1_MAT_s101
                                 + 1.10063 w_k1_MSFT_s101
                                 + 1.00975 w_k1_PG_s101 >= 1
 z_k1_s1020_constraint:          var_1 + z_k1_s1020 + 1.20965 w_k1_ADM_s102
                                 + 1.09227 w_k1_AXP_s102
                                 + 1.63714 w_k1_DIS_s102
                                 + 1.15075 w_k1_FDX_s102
                                 + 1.14039 w_k1_IBM_s102
                                 + 1.05876 w_k1_INTC_s102
                                 + 1.43431 w_k1_KO_s102 + 1.0239 w_k1_MAT_s102
                                 + 1.17484 w_k1_MSFT_s102
                                 + 1.19903 w_k1_PG_s102 >= 1
 z_k1_s1021_constraint:          var_1 + z_k1_s1021 + 0.76548 w_k1_ADM_s102
                                 + 0.741245 w_k1_AXP_s102
                                 + 0.763458 w_k1_DIS_s102
                                 + 0.777535 w_k1_FDX_s102
                                 + 0.987271 w_k1_IBM_s102
                                 + 0.958817 w_k1_INTC_s102
                                 + 0.943157 w_k1_KO_s102
                                 + 1.00076 w_k1_MAT_s102
                                 + 0.875352 w_k1_MSFT_s102
                                 + 0.902644 w_k1_PG_s102 >= 1
 z_k1_s1022_constraint:          var_1 + z_k1_s1022 + 1.03548 w_k1_ADM_s102
                                 + 1.40218 w_k1_AXP_s102
                                 + 1.33195 w_k1_DIS_s102
                                 + 1.14772 w_k1_FDX_s102
                                 + 1.55159 w_k1_IBM_s102
                                 + 1.22144 w_k1_INTC_s102
                                 + 1.36716 w_k1_KO_s102 + 1.62303 w_k1_MAT_s102
                                 + 0.990923 w_k1_MSFT_s102
                                 + 1.07144 w_k1_PG_s102 >= 1
 z_k1_s1023_constraint:          var_1 + z_k1_s1023 + 1.31807 w_k1_ADM_s102
                                 + 1.22303 w_k1_AXP_s102
                                 + 1.82975 w_k1_DIS_s102
                                 + 1.38591 w_k1_FDX_s102
                                 + 1.34646 w_k1_IBM_s102
                                 + 1.15801 w_k1_INTC_s102
                                 + 1.36779 w_k1_KO_s102 + 1.33769 w_k1_MAT_s102
                                 + 1.30019 w_k1_MSFT_s102 + 1.1619 w_k1_PG_s102
                                 >= 1
 z_k1_s1024_constraint:          var_1 + z_k1_s1024 + 0.768745 w_k1_ADM_s102
                                 + 1.01293 w_k1_AXP_s102
                                 + 0.846419 w_k1_DIS_s102
                                 + 0.890982 w_k1_FDX_s102
                                 + 1.38366 w_k1_IBM_s102
                                 + 1.04411 w_k1_INTC_s102
                                 + 1.06355 w_k1_KO_s102 + 1.72898 w_k1_MAT_s102
                                 + 0.925809 w_k1_MSFT_s102
                                 + 0.971029 w_k1_PG_s102 >= 1
 z_k1_s1025_constraint:          var_1 + z_k1_s1025 + 0.99242 w_k1_ADM_s102
                                 + 0.831257 w_k1_AXP_s102
                                 + 0.884999 w_k1_DIS_s102
                                 + 1.01763 w_k1_FDX_s102
                                 + 0.706012 w_k1_IBM_s102
                                 + 1.10445 w_k1_INTC_s102
                                 + 0.796713 w_k1_KO_s102
                                 + 0.605675 w_k1_MAT_s102
                                 + 0.884214 w_k1_MSFT_s102
                                 + 0.831233 w_k1_PG_s102 >= 1
 z_k1_s1026_constraint:          var_1 + z_k1_s1026 + 1.39689 w_k1_ADM_s102
                                 + 1.03033 w_k1_AXP_s102
                                 + 1.34255 w_k1_DIS_s102
                                 + 1.16566 w_k1_FDX_s102
                                 + 0.960693 w_k1_IBM_s102
                                 + 1.11523 w_k1_INTC_s102
                                 + 0.980336 w_k1_KO_s102
                                 + 0.988233 w_k1_MAT_s102
                                 + 0.993486 w_k1_MSFT_s102
                                 + 0.997326 w_k1_PG_s102 >= 1
 z_k1_s1027_constraint:          var_1 + z_k1_s1027 + 0.940356 w_k1_ADM_s102
                                 + 1.12104 w_k1_AXP_s102
                                 + 0.936899 w_k1_DIS_s102
                                 + 0.950149 w_k1_FDX_s102
                                 + 1.19075 w_k1_IBM_s102
                                 + 0.976873 w_k1_INTC_s102
                                 + 0.99854 w_k1_KO_s102
                                 + 0.984478 w_k1_MAT_s102
                                 + 0.922728 w_k1_MSFT_s102
                                 + 0.990394 w_k1_PG_s102 >= 1
 z_k1_s1030_constraint:          var_1 + z_k1_s1030 + 0.916955 w_k1_ADM_s103
                                 + 1.13575 w_k1_AXP_s103
                                 + 0.893621 w_k1_DIS_s103
                                 + 0.922539 w_k1_FDX_s103
                                 + 1.01698 w_k1_IBM_s103
                                 + 0.991661 w_k1_INTC_s103
                                 + 1.0194 w_k1_KO_s103 + 0.937183 w_k1_MAT_s103
                                 + 0.847761 w_k1_MSFT_s103
                                 + 0.924436 w_k1_PG_s103 >= 1
 z_k1_s1031_constraint:          var_1 + z_k1_s1031 + 0.626359 w_k1_ADM_s103
                                 + 0.653907 w_k1_AXP_s103
                                 + 0.609511 w_k1_DIS_s103
                                 + 0.663351 w_k1_FDX_s103
                                 + 0.766313 w_k1_IBM_s103
                                 + 0.806766 w_k1_INTC_s103
                                 + 0.68016 w_k1_KO_s103
                                 + 0.573293 w_k1_MAT_s103
                                 + 0.659975 w_k1_MSFT_s103
                                 + 0.774155 w_k1_PG_s103 >= 1
 z_k1_s1032_constraint:          var_1 + z_k1_s1032 + 1.1362 w_k1_ADM_s103
                                 + 1.28253 w_k1_AXP_s103
                                 + 1.36509 w_k1_DIS_s103 + 1.1082 w_k1_FDX_s103
                                 + 1.35109 w_k1_IBM_s103
                                 + 1.39583 w_k1_INTC_s103
                                 + 1.46182 w_k1_KO_s103 + 1.26843 w_k1_MAT_s103
                                 + 1.18352 w_k1_MSFT_s103
                                 + 1.08981 w_k1_PG_s103 >= 1
 z_k1_s1033_constraint:          var_1 + z_k1_s1033 + 1.10652 w_k1_ADM_s103
                                 + 1.51172 w_k1_AXP_s103
                                 + 1.82042 w_k1_DIS_s103 + 1.3863 w_k1_FDX_s103
                                 + 1.00553 w_k1_IBM_s103
                                 + 1.52462 w_k1_INTC_s103
                                 + 1.54652 w_k1_KO_s103
                                 + 0.917127 w_k1_MAT_s103
                                 + 1.30829 w_k1_MSFT_s103
                                 + 1.23432 w_k1_PG_s103 >= 1
 z_k1_s1034_constraint:          var_1 + z_k1_s1034 + 0.83394 w_k1_ADM_s103
                                 + 0.803396 w_k1_AXP_s103
                                 + 0.763207 w_k1_DIS_s103
                                 + 0.731619 w_k1_FDX_s103
                                 + 1.0228 w_k1_IBM_s103
                                 + 0.933932 w_k1_INTC_s103
                                 + 0.892731 w_k1_KO_s103
                                 + 0.955595 w_k1_MAT_s103
                                 + 0.83234 w_k1_MSFT_s103
                                 + 0.789144 w_k1_PG_s103 >= 1
 z_k1_s1035_constraint:          var_1 + z_k1_s1035 + 0.812005 w_k1_ADM_s103
                                 + 1.02926 w_k1_AXP_s103
                                 + 0.936974 w_k1_DIS_s103
                                 + 0.937473 w_k1_FDX_s103
                                 + 0.779263 w_k1_IBM_s103
                                 + 0.88533 w_k1_INTC_s103
                                 + 0.884957 w_k1_KO_s103
                                 + 0.692796 w_k1_MAT_s103
                                 + 0.945473 w_k1_MSFT_s103
                                 + 0.932595 w_k1_PG_s103 >= 1
 z_k1_s1036_constraint:          var_1 + z_k1_s1036 + 0.806544 w_k1_ADM_s103
                                 + 0.740139 w_k1_AXP_s103
                                 + 0.762629 w_k1_DIS_s103
                                 + 0.837151 w_k1_FDX_s103
                                 + 0.599883 w_k1_IBM_s103
                                 + 0.951014 w_k1_INTC_s103
                                 + 0.80884 w_k1_KO_s103
                                 + 0.649189 w_k1_MAT_s103
                                 + 0.769547 w_k1_MSFT_s103
                                 + 0.764217 w_k1_PG_s103 >= 1
 z_k1_s1037_constraint:          var_1 + z_k1_s1037 + 1.04443 w_k1_ADM_s103
                                 + 1.34934 w_k1_AXP_s103
                                 + 1.21959 w_k1_DIS_s103 + 1.2235 w_k1_FDX_s103
                                 + 0.886428 w_k1_IBM_s103
                                 + 1.23846 w_k1_INTC_s103
                                 + 1.19197 w_k1_KO_s103 + 1.08691 w_k1_MAT_s103
                                 + 1.04374 w_k1_MSFT_s103
                                 + 1.11741 w_k1_PG_s103 >= 1
 z_k1_s1040_constraint:          var_1 + z_k1_s1040 + 0.927391 w_k1_ADM_s104
                                 + 0.957327 w_k1_AXP_s104
                                 + 0.874001 w_k1_DIS_s104
                                 + 0.920975 w_k1_FDX_s104
                                 + 1.07087 w_k1_IBM_s104
                                 + 0.783254 w_k1_INTC_s104
                                 + 0.749529 w_k1_KO_s104 + 1.025 w_k1_MAT_s104
                                 + 0.853279 w_k1_MSFT_s104
                                 + 0.976215 w_k1_PG_s104 >= 1
 z_k1_s1041_constraint:          var_1 + z_k1_s1041 + 1.13762 w_k1_ADM_s104
                                 + 1.40165 w_k1_AXP_s104
                                 + 1.37526 w_k1_DIS_s104
                                 + 1.17138 w_k1_FDX_s104
                                 + 0.957356 w_k1_IBM_s104
                                 + 0.86243 w_k1_INTC_s104 + 1.0103 w_k1_KO_s104
                                 + 0.864847 w_k1_MAT_s104
                                 + 0.991377 w_k1_MSFT_s104
                                 + 1.09403 w_k1_PG_s104 >= 1
 z_k1_s1042_constraint:          var_1 + z_k1_s1042 + 1.10329 w_k1_ADM_s104
                                 + 1.11705 w_k1_AXP_s104
                                 + 1.13292 w_k1_DIS_s104 + 1.124 w_k1_FDX_s104
                                 + 1.25594 w_k1_IBM_s104
                                 + 0.853318 w_k1_INTC_s104
                                 + 1.1847 w_k1_KO_s104 + 0.986308 w_k1_MAT_s104
                                 + 0.85558 w_k1_MSFT_s104
                                 + 1.06687 w_k1_PG_s104 >= 1
 z_k1_s1043_constraint:          var_1 + z_k1_s1043 + 1.01682 w_k1_ADM_s104
                                 + 0.901567 w_k1_AXP_s104 + 1.174 w_k1_DIS_s104
                                 + 0.967563 w_k1_FDX_s104
                                 + 1.29636 w_k1_IBM_s104
                                 + 0.780623 w_k1_INTC_s104
                                 + 1.02989 w_k1_KO_s104 + 1.19297 w_k1_MAT_s104
                                 + 0.883144 w_k1_MSFT_s104
                                 + 1.05438 w_k1_PG_s104 >= 1
 z_k1_s1044_constraint:          var_1 + z_k1_s1044 + 1.43804 w_k1_ADM_s104
                                 + 1.44367 w_k1_AXP_s104
                                 + 1.89416 w_k1_DIS_s104
                                 + 1.47309 w_k1_FDX_s104
                                 + 1.17012 w_k1_IBM_s104
                                 + 1.12301 w_k1_INTC_s104
                                 + 1.37346 w_k1_KO_s104 + 1.13885 w_k1_MAT_s104
                                 + 1.31803 w_k1_MSFT_s104
                                 + 1.36961 w_k1_PG_s104 >= 1
 z_k1_s1045_constraint:          var_1 + z_k1_s1045 + 0.72568 w_k1_ADM_s104
                                 + 0.689614 w_k1_AXP_s104
                                 + 0.674688 w_k1_DIS_s104
                                 + 0.750544 w_k1_FDX_s104
                                 + 0.928463 w_k1_IBM_s104
                                 + 0.822728 w_k1_INTC_s104
                                 + 0.829111 w_k1_KO_s104
                                 + 0.832305 w_k1_MAT_s104
                                 + 0.708001 w_k1_MSFT_s104
                                 + 0.915342 w_k1_PG_s104 >= 1
 z_k1_s1046_constraint:          var_1 + z_k1_s1046 + 0.974142 w_k1_ADM_s104
                                 + 1.30002 w_k1_AXP_s104 + 1.6247 w_k1_DIS_s104
                                 + 1.25946 w_k1_FDX_s104
                                 + 1.29836 w_k1_IBM_s104
                                 + 1.1482 w_k1_INTC_s104 + 1.23633 w_k1_KO_s104
                                 + 1.40776 w_k1_MAT_s104
                                 + 1.1932 w_k1_MSFT_s104 + 1.07592 w_k1_PG_s104
                                 >= 1
 z_k1_s1047_constraint:          var_1 + z_k1_s1047 + 0.994973 w_k1_ADM_s104
                                 + 0.813938 w_k1_AXP_s104
                                 + 0.801488 w_k1_DIS_s104
                                 + 0.864103 w_k1_FDX_s104
                                 + 0.688071 w_k1_IBM_s104
                                 + 0.84977 w_k1_INTC_s104
                                 + 0.713042 w_k1_KO_s104
                                 + 0.561742 w_k1_MAT_s104
                                 + 0.838078 w_k1_MSFT_s104
                                 + 0.888094 w_k1_PG_s104 >= 1
 z_k1_s1050_constraint:          var_1 + z_k1_s1050 + 0.97771 w_k1_ADM_s105
                                 + 0.962959 w_k1_AXP_s105
                                 + 0.890515 w_k1_DIS_s105
                                 + 0.900763 w_k1_FDX_s105
                                 + 1.13684 w_k1_IBM_s105
                                 + 1.24668 w_k1_INTC_s105
                                 + 1.18551 w_k1_KO_s105 + 1.69627 w_k1_MAT_s105
                                 + 0.925722 w_k1_MSFT_s105
                                 + 0.929792 w_k1_PG_s105 >= 1
 z_k1_s1051_constraint:          var_1 + z_k1_s1051 + 1.79548 w_k1_ADM_s105
                                 + 1.60796 w_k1_AXP_s105
                                 + 1.51286 w_k1_DIS_s105 + 1.3161 w_k1_FDX_s105
                                 + 1.0444 w_k1_IBM_s105
                                 + 1.33693 w_k1_INTC_s105 + 1.1756 w_k1_KO_s105
                                 + 1.25784 w_k1_MAT_s105
                                 + 1.18041 w_k1_MSFT_s105
                                 + 1.23759 w_k1_PG_s105 >= 1
 z_k1_s1052_constraint:          var_1 + z_k1_s1052 + 1.06867 w_k1_ADM_s105
                                 + 1.64106 w_k1_AXP_s105
                                 + 1.71859 w_k1_DIS_s105
                                 + 1.26586 w_k1_FDX_s105
                                 + 1.60154 w_k1_IBM_s105
                                 + 1.40485 w_k1_INTC_s105 + 1.493 w_k1_KO_s105
                                 + 2.25401 w_k1_MAT_s105
                                 + 1.25364 w_k1_MSFT_s105
                                 + 1.18253 w_k1_PG_s105 >= 1
 z_k1_s1053_constraint:          var_1 + z_k1_s1053 + 0.788722 w_k1_ADM_s105
                                 + 1.01058 w_k1_AXP_s105
                                 + 0.992266 w_k1_DIS_s105
                                 + 0.84566 w_k1_FDX_s105
                                 + 1.24534 w_k1_IBM_s105
                                 + 0.983167 w_k1_INTC_s105
                                 + 1.15888 w_k1_KO_s105 + 1.09894 w_k1_MAT_s105
                                 + 0.693983 w_k1_MSFT_s105
                                 + 0.804073 w_k1_PG_s105 >= 1
 z_k1_s1054_constraint:          var_1 + z_k1_s1054 + 0.784863 w_k1_ADM_s105
                                 + 0.687375 w_k1_AXP_s105
                                 + 0.607029 w_k1_DIS_s105
                                 + 0.709001 w_k1_FDX_s105
                                 + 0.664672 w_k1_IBM_s105
                                 + 0.83441 w_k1_INTC_s105
                                 + 0.666399 w_k1_KO_s105
                                 + 0.576895 w_k1_MAT_s105
                                 + 0.617949 w_k1_MSFT_s105
                                 + 0.735678 w_k1_PG_s105 >= 1
 z_k1_s1055_constraint:          var_1 + z_k1_s1055 + 0.8201 w_k1_ADM_s105
                                 + 0.613556 w_k1_AXP_s105
                                 + 0.655727 w_k1_DIS_s105
                                 + 0.655117 w_k1_FDX_s105
                                 + 0.942083 w_k1_IBM_s105
                                 + 0.816357 w_k1_INTC_s105
                                 + 0.748214 w_k1_KO_s105
                                 + 0.735928 w_k1_MAT_s105
                                 + 0.654346 w_k1_MSFT_s105
                                 + 0.703019 w_k1_PG_s105 >= 1
 z_k1_s1056_constraint:          var_1 + z_k1_s1056 + 1.13369 w_k1_ADM_s105
                                 + 0.946066 w_k1_AXP_s105
                                 + 1.02405 w_k1_DIS_s105
                                 + 1.01915 w_k1_FDX_s105
                                 + 0.741108 w_k1_IBM_s105
                                 + 1.20631 w_k1_INTC_s105
                                 + 0.85697 w_k1_KO_s105
                                 + 0.563786 w_k1_MAT_s105
                                 + 0.9766 w_k1_MSFT_s105
                                 + 0.800571 w_k1_PG_s105 >= 1
 z_k1_s1057_constraint:          var_1 + z_k1_s1057 + 0.651103 w_k1_ADM_s105
                                 + 1.16786 w_k1_AXP_s105
                                 + 0.724017 w_k1_DIS_s105
                                 + 0.883946 w_k1_FDX_s105
                                 + 0.782207 w_k1_IBM_s105
                                 + 0.830299 w_k1_INTC_s105
                                 + 0.754982 w_k1_KO_s105
                                 + 1.08578 w_k1_MAT_s105
                                 + 0.775501 w_k1_MSFT_s105
                                 + 0.733741 w_k1_PG_s105 >= 1
 z_k1_s1060_constraint:          var_1 + z_k1_s1060 + 1.09344 w_k1_ADM_s106
                                 + 1.38064 w_k1_AXP_s106
                                 + 1.02452 w_k1_DIS_s106 + 1.2064 w_k1_FDX_s106
                                 + 0.628712 w_k1_IBM_s106
                                 + 1.18464 w_k1_INTC_s106
                                 + 0.846337 w_k1_KO_s106
                                 + 0.894845 w_k1_MAT_s106
                                 + 1.03923 w_k1_MSFT_s106
                                 + 0.957626 w_k1_PG_s106 >= 1
 z_k1_s1061_constraint:          var_1 + z_k1_s1061 + 1.24591 w_k1_ADM_s106
                                 + 0.98876 w_k1_AXP_s106
                                 + 1.45558 w_k1_DIS_s106
                                 + 1.21094 w_k1_FDX_s106
                                 + 0.989601 w_k1_IBM_s106
                                 + 1.1653 w_k1_INTC_s106 + 1.23773 w_k1_KO_s106
                                 + 0.884209 w_k1_MAT_s106
                                 + 1.21541 w_k1_MSFT_s106
                                 + 1.08393 w_k1_PG_s106 >= 1
 z_k1_s1062_constraint:          var_1 + z_k1_s1062 + 1.07925 w_k1_ADM_s106
                                 + 1.39515 w_k1_AXP_s106
                                 + 0.789883 w_k1_DIS_s106
                                 + 1.0294 w_k1_FDX_s106 + 1.38332 w_k1_IBM_s106
                                 + 1.15108 w_k1_INTC_s106
                                 + 1.02172 w_k1_KO_s106 + 1.73908 w_k1_MAT_s106
                                 + 0.906779 w_k1_MSFT_s106
                                 + 1.10162 w_k1_PG_s106 >= 1
 z_k1_s1063_constraint:          var_1 + z_k1_s1063 + 1.37528 w_k1_ADM_s106
                                 + 1.43584 w_k1_AXP_s106
                                 + 1.55641 w_k1_DIS_s106
                                 + 1.29339 w_k1_FDX_s106
                                 + 1.56191 w_k1_IBM_s106
                                 + 1.34151 w_k1_INTC_s106
                                 + 1.25787 w_k1_KO_s106 + 1.05493 w_k1_MAT_s106
                                 + 1.121 w_k1_MSFT_s106 + 1.08798 w_k1_PG_s106
                                 >= 1
 z_k1_s1064_constraint:          var_1 + z_k1_s1064 + 1.0378 w_k1_ADM_s106
                                 + 1.17366 w_k1_AXP_s106
                                 + 0.999771 w_k1_DIS_s106
                                 + 0.95787 w_k1_FDX_s106
                                 + 1.03492 w_k1_IBM_s106
                                 + 1.07223 w_k1_INTC_s106
                                 + 1.08515 w_k1_KO_s106 + 1.35537 w_k1_MAT_s106
                                 + 0.963274 w_k1_MSFT_s106
                                 + 0.96897 w_k1_PG_s106 >= 1
 z_k1_s1065_constraint:          var_1 + z_k1_s1065 + 0.997196 w_k1_ADM_s106
                                 + 0.906463 w_k1_AXP_s106
                                 + 0.887735 w_k1_DIS_s106
                                 + 0.806053 w_k1_FDX_s106
                                 + 1.11679 w_k1_IBM_s106
                                 + 1.00191 w_k1_INTC_s106
                                 + 1.11349 w_k1_KO_s106
                                 + 0.991656 w_k1_MAT_s106
                                 + 0.880333 w_k1_MSFT_s106
                                 + 0.921909 w_k1_PG_s106 >= 1
 z_k1_s1066_constraint:          var_1 + z_k1_s1066 + 0.871961 w_k1_ADM_s106
                                 + 0.789792 w_k1_AXP_s106
                                 + 0.558453 w_k1_DIS_s106
                                 + 0.769328 w_k1_FDX_s106
                                 + 0.769484 w_k1_IBM_s106
                                 + 0.968781 w_k1_INTC_s106
                                 + 0.714179 w_k1_KO_s106
                                 + 0.755436 w_k1_MAT_s106
                                 + 0.69855 w_k1_MSFT_s106
                                 + 0.80634 w_k1_PG_s106 >= 1
 z_k1_s1067_constraint:          var_1 + z_k1_s1067 + 1.18033 w_k1_ADM_s106
                                 + 1.17272 w_k1_AXP_s106
                                 + 0.810791 w_k1_DIS_s106
                                 + 1.07762 w_k1_FDX_s106
                                 + 0.952736 w_k1_IBM_s106
                                 + 1.06906 w_k1_INTC_s106
                                 + 0.899126 w_k1_KO_s106
                                 + 1.17553 w_k1_MAT_s106
                                 + 1.01652 w_k1_MSFT_s106
                                 + 0.979945 w_k1_PG_s106 >= 1
 z_k1_s1070_constraint:          var_1 + z_k1_s1070 + 0.894417 w_k1_ADM_s107
                                 + 1.55237 w_k1_AXP_s107
                                 + 1.56223 w_k1_DIS_s107
                                 + 1.14961 w_k1_FDX_s107
                                 + 1.60099 w_k1_IBM_s107
                                 + 1.09873 w_k1_INTC_s107
                                 + 1.36899 w_k1_KO_s107 + 1.51022 w_k1_MAT_s107
                                 + 1.15184 w_k1_MSFT_s107
                                 + 1.28334 w_k1_PG_s107 >= 1
 z_k1_s1071_constraint:          var_1 + z_k1_s1071 + 1.36748 w_k1_ADM_s107
                                 + 1.29144 w_k1_AXP_s107
                                 + 1.44496 w_k1_DIS_s107
                                 + 1.33662 w_k1_FDX_s107
                                 + 0.940647 w_k1_IBM_s107
                                 + 1.16592 w_k1_INTC_s107
                                 + 1.12383 w_k1_KO_s107
                                 + 0.759222 w_k1_MAT_s107
                                 + 1.25302 w_k1_MSFT_s107
                                 + 1.09518 w_k1_PG_s107 >= 1
 z_k1_s1072_constraint:          var_1 + z_k1_s1072 + 1.07673 w_k1_ADM_s107
                                 + 1.04953 w_k1_AXP_s107
                                 + 1.01042 w_k1_DIS_s107
                                 + 1.03164 w_k1_FDX_s107
                                 + 0.730577 w_k1_IBM_s107
                                 + 0.867477 w_k1_INTC_s107
                                 + 0.840023 w_k1_KO_s107
                                 + 0.560141 w_k1_MAT_s107
                                 + 0.861318 w_k1_MSFT_s107
                                 + 0.903747 w_k1_PG_s107 >= 1
 z_k1_s1073_constraint:          var_1 + z_k1_s1073 + 0.825572 w_k1_ADM_s107
                                 + 0.943128 w_k1_AXP_s107
                                 + 0.820202 w_k1_DIS_s107
                                 + 0.83672 w_k1_FDX_s107
                                 + 0.993134 w_k1_IBM_s107
                                 + 0.839501 w_k1_INTC_s107
                                 + 0.939349 w_k1_KO_s107
                                 + 0.805579 w_k1_MAT_s107
                                 + 0.801167 w_k1_MSFT_s107
                                 + 0.901247 w_k1_PG_s107 >= 1
 z_k1_s1074_constraint:          var_1 + z_k1_s1074 + 1.53858 w_k1_ADM_s107
                                 + 1.57482 w_k1_AXP_s107
                                 + 1.23957 w_k1_DIS_s107
                                 + 1.23908 w_k1_FDX_s107
                                 + 1.01022 w_k1_IBM_s107
                                 + 0.950725 w_k1_INTC_s107
                                 + 0.873703 w_k1_KO_s107
                                 + 1.32357 w_k1_MAT_s107
                                 + 1.05266 w_k1_MSFT_s107 + 1.1598 w_k1_PG_s107
                                 >= 1
 z_k1_s1075_constraint:          var_1 + z_k1_s1075 + 1.07408 w_k1_ADM_s107
                                 + 0.940969 w_k1_AXP_s107
                                 + 0.946911 w_k1_DIS_s107
                                 + 0.758307 w_k1_FDX_s107
                                 + 1.45388 w_k1_IBM_s107
                                 + 0.902361 w_k1_INTC_s107
                                 + 1.05529 w_k1_KO_s107
                                 + 0.951895 w_k1_MAT_s107
                                 + 0.888536 w_k1_MSFT_s107
                                 + 0.964818 w_k1_PG_s107 >= 1
 z_k1_s1076_constraint:          var_1 + z_k1_s1076 + 0.847949 w_k1_ADM_s107
                                 + 0.777284 w_k1_AXP_s107
                                 + 0.639441 w_k1_DIS_s107
                                 + 0.764114 w_k1_FDX_s107
                                 + 0.714875 w_k1_IBM_s107
                                 + 0.793423 w_k1_INTC_s107
                                 + 0.706776 w_k1_KO_s107
                                 + 0.587183 w_k1_MAT_s107
                                 + 0.702442 w_k1_MSFT_s107
                                 + 0.885232 w_k1_PG_s107 >= 1
 z_k1_s1077_constraint:          var_1 + z_k1_s1077 + 1.26919 w_k1_ADM_s107
                                 + 1.21126 w_k1_AXP_s107
                                 + 1.11597 w_k1_DIS_s107
                                 + 1.08925 w_k1_FDX_s107
                                 + 1.24586 w_k1_IBM_s107
                                 + 1.02504 w_k1_INTC_s107
                                 + 1.21649 w_k1_KO_s107 + 1.46547 w_k1_MAT_s107
                                 + 1.01498 w_k1_MSFT_s107 + 1.1883 w_k1_PG_s107
                                 >= 1
 z_k1_s1100_constraint:          var_1 + z_k1_s1100 + 1.04184 w_k1_ADM_s110
                                 + 1.09967 w_k1_AXP_s110
                                 + 1.17956 w_k1_DIS_s110
                                 + 1.01595 w_k1_FDX_s110
                                 + 1.48747 w_k1_IBM_s110
                                 + 0.96917 w_k1_INTC_s110
                                 + 1.17238 w_k1_KO_s110 + 1.47569 w_k1_MAT_s110
                                 + 1.01574 w_k1_MSFT_s110
                                 + 1.05309 w_k1_PG_s110 >= 1
 z_k1_s1101_constraint:          var_1 + z_k1_s1101 + 1.30627 w_k1_ADM_s110
                                 + 0.848773 w_k1_AXP_s110
                                 + 0.980546 w_k1_DIS_s110
                                 + 1.08069 w_k1_FDX_s110
                                 + 0.635749 w_k1_IBM_s110
                                 + 0.787156 w_k1_INTC_s110
                                 + 0.828602 w_k1_KO_s110
                                 + 0.698595 w_k1_MAT_s110
                                 + 1.06676 w_k1_MSFT_s110
                                 + 0.806755 w_k1_PG_s110 >= 1
 z_k1_s1102_constraint:          var_1 + z_k1_s1102 + 0.770338 w_k1_ADM_s110
                                 + 0.923876 w_k1_AXP_s110
                                 + 0.88532 w_k1_DIS_s110
                                 + 0.973266 w_k1_FDX_s110
                                 + 1.12878 w_k1_IBM_s110
                                 + 0.847211 w_k1_INTC_s110
                                 + 0.954049 w_k1_KO_s110
                                 + 1.21366 w_k1_MAT_s110
                                 + 1.06654 w_k1_MSFT_s110
                                 + 0.900051 w_k1_PG_s110 >= 1
 z_k1_s1103_constraint:          var_1 + z_k1_s1103 + 0.73413 w_k1_ADM_s110
                                 + 0.741457 w_k1_AXP_s110
                                 + 0.743789 w_k1_DIS_s110
                                 + 0.774898 w_k1_FDX_s110
                                 + 0.965815 w_k1_IBM_s110
                                 + 0.970435 w_k1_INTC_s110
                                 + 0.856256 w_k1_KO_s110
                                 + 1.05455 w_k1_MAT_s110
                                 + 0.976897 w_k1_MSFT_s110
                                 + 0.840486 w_k1_PG_s110 >= 1
 z_k1_s1104_constraint:          var_1 + z_k1_s1104 + 1.28859 w_k1_ADM_s110
                                 + 1.23896 w_k1_AXP_s110
                                 + 1.60007 w_k1_DIS_s110
                                 + 1.18004 w_k1_FDX_s110
                                 + 1.44787 w_k1_IBM_s110
                                 + 1.06536 w_k1_INTC_s110
                                 + 1.43731 w_k1_KO_s110 + 1.45317 w_k1_MAT_s110
                                 + 1.30747 w_k1_MSFT_s110
                                 + 1.12531 w_k1_PG_s110 >= 1
 z_k1_s1105_constraint:          var_1 + z_k1_s1105 + 1.58961 w_k1_ADM_s110
                                 + 1.60456 w_k1_AXP_s110
                                 + 1.12281 w_k1_DIS_s110
                                 + 1.23636 w_k1_FDX_s110
                                 + 1.15848 w_k1_IBM_s110
                                 + 1.13582 w_k1_INTC_s110
                                 + 1.14663 w_k1_KO_s110 + 1.42527 w_k1_MAT_s110
                                 + 1.17627 w_k1_MSFT_s110
                                 + 1.01801 w_k1_PG_s110 >= 1
 z_k1_s1106_constraint:          var_1 + z_k1_s1106 + 0.8136 w_k1_ADM_s110
                                 + 0.855477 w_k1_AXP_s110
                                 + 0.844883 w_k1_DIS_s110
                                 + 0.90178 w_k1_FDX_s110
                                 + 0.700512 w_k1_IBM_s110
                                 + 0.914245 w_k1_INTC_s110
                                 + 0.826885 w_k1_KO_s110
                                 + 0.618759 w_k1_MAT_s110
                                 + 0.970799 w_k1_MSFT_s110
                                 + 0.813993 w_k1_PG_s110 >= 1
 z_k1_s1107_constraint:          var_1 + z_k1_s1107 + 0.941034 w_k1_ADM_s110
                                 + 1.25282 w_k1_AXP_s110
                                 + 1.32911 w_k1_DIS_s110
                                 + 1.28096 w_k1_FDX_s110
                                 + 0.838167 w_k1_IBM_s110
                                 + 1.17071 w_k1_INTC_s110
                                 + 1.00052 w_k1_KO_s110 + 0.85073 w_k1_MAT_s110
                                 + 1.41217 w_k1_MSFT_s110
                                 + 1.02982 w_k1_PG_s110 >= 1
 z_k1_s1110_constraint:          var_1 + z_k1_s1110 + 0.911544 w_k1_ADM_s111
                                 + 1.81372 w_k1_AXP_s111
                                 + 1.62887 w_k1_DIS_s111
                                 + 1.41888 w_k1_FDX_s111
                                 + 1.20356 w_k1_IBM_s111
                                 + 1.51263 w_k1_INTC_s111
                                 + 1.36375 w_k1_KO_s111 + 1.90815 w_k1_MAT_s111
                                 + 1.4605 w_k1_MSFT_s111 + 1.26809 w_k1_PG_s111
                                 >= 1
 z_k1_s1111_constraint:          var_1 + z_k1_s1111 + 0.800389 w_k1_ADM_s111
                                 + 1.13185 w_k1_AXP_s111
                                 + 1.22325 w_k1_DIS_s111
                                 + 1.20981 w_k1_FDX_s111
                                 + 0.809972 w_k1_IBM_s111
                                 + 1.37596 w_k1_INTC_s111
                                 + 0.914183 w_k1_KO_s111
                                 + 0.964179 w_k1_MAT_s111
                                 + 1.24701 w_k1_MSFT_s111 + 1.0402 w_k1_PG_s111
                                 >= 1
 z_k1_s1112_constraint:          var_1 + z_k1_s1112 + 0.769599 w_k1_ADM_s111
                                 + 1.03949 w_k1_AXP_s111
                                 + 0.964865 w_k1_DIS_s111
                                 + 0.870396 w_k1_FDX_s111
                                 + 1.32374 w_k1_IBM_s111
                                 + 0.984543 w_k1_INTC_s111
                                 + 1.03821 w_k1_KO_s111 + 1.23675 w_k1_MAT_s111
                                 + 0.990615 w_k1_MSFT_s111
                                 + 0.911688 w_k1_PG_s111 >= 1
 z_k1_s1113_constraint:          var_1 + z_k1_s1113 + 0.723752 w_k1_ADM_s111
                                 + 1.25552 w_k1_AXP_s111
                                 + 1.36908 w_k1_DIS_s111
                                 + 1.16151 w_k1_FDX_s111
                                 + 1.59616 w_k1_IBM_s111
                                 + 1.37265 w_k1_INTC_s111
                                 + 1.33006 w_k1_KO_s111 + 2.02465 w_k1_MAT_s111
                                 + 1.29948 w_k1_MSFT_s111
                                 + 1.03539 w_k1_PG_s111 >= 1
 z_k1_s1114_constraint:          var_1 + z_k1_s1114 + 0.835878 w_k1_ADM_s111
                                 + 0.855607 w_k1_AXP_s111
                                 + 0.687183 w_k1_DIS_s111
                                 + 0.777101 w_k1_FDX_s111
                                 + 0.817037 w_k1_IBM_s111
                                 + 1.00017 w_k1_INTC_s111
                                 + 0.727193 w_k1_KO_s111
                                 + 0.691332 w_k1_MAT_s111
                                 + 0.902691 w_k1_MSFT_s111
                                 + 0.780095 w_k1_PG_s111 >= 1
 z_k1_s1115_constraint:          var_1 + z_k1_s1115 + 0.762607 w_k1_ADM_s111
                                 + 1.16964 w_k1_AXP_s111
                                 + 1.28717 w_k1_DIS_s111
                                 + 1.10226 w_k1_FDX_s111
                                 + 1.02874 w_k1_IBM_s111
                                 + 1.10773 w_k1_INTC_s111
                                 + 1.07689 w_k1_KO_s111 + 1.20014 w_k1_MAT_s111
                                 + 1.19822 w_k1_MSFT_s111
                                 + 0.96938 w_k1_PG_s111 >= 1
 z_k1_s1116_constraint:          var_1 + z_k1_s1116 + 1.05212 w_k1_ADM_s111
                                 + 0.994528 w_k1_AXP_s111
                                 + 0.774838 w_k1_DIS_s111
                                 + 0.834611 w_k1_FDX_s111
                                 + 1.13587 w_k1_IBM_s111
                                 + 0.929819 w_k1_INTC_s111
                                 + 0.826729 w_k1_KO_s111
                                 + 1.13522 w_k1_MAT_s111
                                 + 0.908189 w_k1_MSFT_s111
                                 + 1.00161 w_k1_PG_s111 >= 1
 z_k1_s1117_constraint:          var_1 + z_k1_s1117 + 0.972941 w_k1_ADM_s111
                                 + 1.23513 w_k1_AXP_s111
                                 + 1.30417 w_k1_DIS_s111
                                 + 1.06739 w_k1_FDX_s111
                                 + 1.21718 w_k1_IBM_s111
                                 + 0.994727 w_k1_INTC_s111
                                 + 1.09098 w_k1_KO_s111 + 1.52418 w_k1_MAT_s111
                                 + 1.18429 w_k1_MSFT_s111
                                 + 1.07803 w_k1_PG_s111 >= 1
 z_k1_s1120_constraint:          var_1 + z_k1_s1120 + 1.10224 w_k1_ADM_s112
                                 + 0.752922 w_k1_AXP_s112
                                 + 1.05747 w_k1_DIS_s112
                                 + 1.06608 w_k1_FDX_s112
                                 + 0.626867 w_k1_IBM_s112
                                 + 0.900287 w_k1_INTC_s112
                                 + 0.796812 w_k1_KO_s112
                                 + 0.459399 w_k1_MAT_s112
                                 + 0.977183 w_k1_MSFT_s112
                                 + 0.938614 w_k1_PG_s112 >= 1
 z_k1_s1121_constraint:          var_1 + z_k1_s1121 + 1.04697 w_k1_ADM_s112
                                 + 0.781992 w_k1_AXP_s112
                                 + 0.908559 w_k1_DIS_s112
                                 + 1.09219 w_k1_FDX_s112
                                 + 0.735145 w_k1_IBM_s112
                                 + 0.998142 w_k1_INTC_s112
                                 + 0.865847 w_k1_KO_s112
                                 + 0.685055 w_k1_MAT_s112
                                 + 1.17593 w_k1_MSFT_s112
                                 + 1.11424 w_k1_PG_s112 >= 1
 z_k1_s1122_constraint:          var_1 + z_k1_s1122 + 0.938886 w_k1_ADM_s112
                                 + 1.00743 w_k1_AXP_s112
                                 + 0.948559 w_k1_DIS_s112
                                 + 0.985653 w_k1_FDX_s112
                                 + 1.41878 w_k1_IBM_s112
                                 + 0.876228 w_k1_INTC_s112
                                 + 1.03555 w_k1_KO_s112 + 1.08471 w_k1_MAT_s112
                                 + 0.897056 w_k1_MSFT_s112
                                 + 1.02093 w_k1_PG_s112 >= 1
 z_k1_s1123_constraint:          var_1 + z_k1_s1123 + 0.804009 w_k1_ADM_s112
                                 + 0.524047 w_k1_AXP_s112
                                 + 0.608964 w_k1_DIS_s112
                                 + 0.648469 w_k1_FDX_s112
                                 + 0.927492 w_k1_IBM_s112
                                 + 0.749593 w_k1_INTC_s112
                                 + 0.912567 w_k1_KO_s112
                                 + 0.586085 w_k1_MAT_s112
                                 + 0.805636 w_k1_MSFT_s112
                                 + 0.913053 w_k1_PG_s112 >= 1
 z_k1_s1124_constraint:          var_1 + z_k1_s1124 + 0.884096 w_k1_ADM_s112
                                 + 0.882954 w_k1_AXP_s112
                                 + 0.85403 w_k1_DIS_s112
                                 + 0.864612 w_k1_FDX_s112
                                 + 0.963902 w_k1_IBM_s112
                                 + 0.798146 w_k1_INTC_s112
                                 + 0.991346 w_k1_KO_s112
                                 + 0.810628 w_k1_MAT_s112
                                 + 0.950956 w_k1_MSFT_s112
                                 + 1.04267 w_k1_PG_s112 >= 1
 z_k1_s1125_constraint:          var_1 + z_k1_s1125 + 1.09648 w_k1_ADM_s112
                                 + 1.18415 w_k1_AXP_s112 + 1.3142 w_k1_DIS_s112
                                 + 1.19168 w_k1_FDX_s112
                                 + 1.08946 w_k1_IBM_s112
                                 + 1.07259 w_k1_INTC_s112
                                 + 1.36508 w_k1_KO_s112 + 1.06091 w_k1_MAT_s112
                                 + 1.17278 w_k1_MSFT_s112
                                 + 1.27006 w_k1_PG_s112 >= 1
 z_k1_s1126_constraint:          var_1 + z_k1_s1126 + 1.03825 w_k1_ADM_s112
                                 + 1.26582 w_k1_AXP_s112
                                 + 1.08051 w_k1_DIS_s112
                                 + 1.10523 w_k1_FDX_s112
                                 + 0.730518 w_k1_IBM_s112
                                 + 1.16478 w_k1_INTC_s112
                                 + 0.905208 w_k1_KO_s112
                                 + 0.907866 w_k1_MAT_s112
                                 + 0.994897 w_k1_MSFT_s112
                                 + 1.07492 w_k1_PG_s112 >= 1
 z_k1_s1127_constraint:          var_1 + z_k1_s1127 + 0.676664 w_k1_ADM_s112
                                 + 0.661062 w_k1_AXP_s112
                                 + 0.678558 w_k1_DIS_s112
                                 + 0.75037 w_k1_FDX_s112
                                 + 0.719915 w_k1_IBM_s112
                                 + 0.885681 w_k1_INTC_s112
                                 + 0.808959 w_k1_KO_s112
                                 + 0.498163 w_k1_MAT_s112
                                 + 0.875563 w_k1_MSFT_s112
                                 + 0.980814 w_k1_PG_s112 >= 1
 z_k1_s1130_constraint:          var_1 + z_k1_s1130 + 0.989893 w_k1_ADM_s113
                                 + 1.4252 w_k1_AXP_s113 + 1.09766 w_k1_DIS_s113
                                 + 1.10471 w_k1_FDX_s113
                                 + 1.16354 w_k1_IBM_s113
                                 + 1.30909 w_k1_INTC_s113
                                 + 0.980162 w_k1_KO_s113
                                 + 1.16516 w_k1_MAT_s113
                                 + 1.00467 w_k1_MSFT_s113
                                 + 1.14324 w_k1_PG_s113 >= 1
 z_k1_s1131_constraint:          var_1 + z_k1_s1131 + 1.20818 w_k1_ADM_s113
                                 + 0.904284 w_k1_AXP_s113
                                 + 1.32579 w_k1_DIS_s113
                                 + 1.08027 w_k1_FDX_s113
                                 + 1.44375 w_k1_IBM_s113
                                 + 1.09488 w_k1_INTC_s113
                                 + 1.28271 w_k1_KO_s113 + 1.23602 w_k1_MAT_s113
                                 + 1.04173 w_k1_MSFT_s113
                                 + 1.09802 w_k1_PG_s113 >= 1
 z_k1_s1132_constraint:          var_1 + z_k1_s1132 + 0.767982 w_k1_ADM_s113
                                 + 0.880232 w_k1_AXP_s113
                                 + 0.737462 w_k1_DIS_s113
                                 + 0.829599 w_k1_FDX_s113
                                 + 0.759339 w_k1_IBM_s113
                                 + 0.886114 w_k1_INTC_s113
                                 + 0.74525 w_k1_KO_s113
                                 + 0.653178 w_k1_MAT_s113
                                 + 0.6677 w_k1_MSFT_s113
                                 + 0.897496 w_k1_PG_s113 >= 1
 z_k1_s1133_constraint:          var_1 + z_k1_s1133 + 0.880028 w_k1_ADM_s113
                                 + 0.924659 w_k1_AXP_s113
                                 + 0.891453 w_k1_DIS_s113
                                 + 0.792105 w_k1_FDX_s113
                                 + 1.63117 w_k1_IBM_s113
                                 + 0.895079 w_k1_INTC_s113
                                 + 1.21294 w_k1_KO_s113 + 1.25137 w_k1_MAT_s113
                                 + 0.844108 w_k1_MSFT_s113
                                 + 0.972064 w_k1_PG_s113 >= 1
 z_k1_s1134_constraint:          var_1 + z_k1_s1134 + 0.695286 w_k1_ADM_s113
                                 + 0.684452 w_k1_AXP_s113
                                 + 0.602597 w_k1_DIS_s113
                                 + 0.736083 w_k1_FDX_s113
                                 + 1.02893 w_k1_IBM_s113
                                 + 0.772269 w_k1_INTC_s113
                                 + 0.854529 w_k1_KO_s113
                                 + 0.990603 w_k1_MAT_s113
                                 + 0.747869 w_k1_MSFT_s113
                                 + 0.909674 w_k1_PG_s113 >= 1
 z_k1_s1135_constraint:          var_1 + z_k1_s1135 + 1.42297 w_k1_ADM_s113
                                 + 1.80748 w_k1_AXP_s113
                                 + 1.47189 w_k1_DIS_s113
                                 + 1.29761 w_k1_FDX_s113 + 1.629 w_k1_IBM_s113
                                 + 1.488 w_k1_INTC_s113 + 1.30005 w_k1_KO_s113
                                 + 1.28936 w_k1_MAT_s113
                                 + 1.23433 w_k1_MSFT_s113
                                 + 1.35945 w_k1_PG_s113 >= 1
 z_k1_s1136_constraint:          var_1 + z_k1_s1136 + 1.39281 w_k1_ADM_s113
                                 + 1.06137 w_k1_AXP_s113
                                 + 1.29521 w_k1_DIS_s113
                                 + 1.30096 w_k1_FDX_s113
                                 + 0.902224 w_k1_IBM_s113
                                 + 1.16218 w_k1_INTC_s113
                                 + 1.10958 w_k1_KO_s113
                                 + 0.718395 w_k1_MAT_s113
                                 + 1.23284 w_k1_MSFT_s113
                                 + 1.21603 w_k1_PG_s113 >= 1
 z_k1_s1137_constraint:          var_1 + z_k1_s1137 + 1.16114 w_k1_ADM_s113
                                 + 0.966126 w_k1_AXP_s113
                                 + 0.996499 w_k1_DIS_s113
                                 + 0.988994 w_k1_FDX_s113
                                 + 0.884567 w_k1_IBM_s113
                                 + 0.883326 w_k1_INTC_s113
                                 + 0.897634 w_k1_KO_s113
                                 + 0.921233 w_k1_MAT_s113
                                 + 0.880334 w_k1_MSFT_s113
                                 + 0.964952 w_k1_PG_s113 >= 1
 z_k1_s1140_constraint:          var_1 + z_k1_s1140 + 1.34321 w_k1_ADM_s114
                                 + 1.32048 w_k1_AXP_s114
                                 + 1.35419 w_k1_DIS_s114 + 1.2472 w_k1_FDX_s114
                                 + 1.06795 w_k1_IBM_s114
                                 + 1.3244 w_k1_INTC_s114 + 1.12738 w_k1_KO_s114
                                 + 1.19212 w_k1_MAT_s114
                                 + 1.37975 w_k1_MSFT_s114
                                 + 1.06152 w_k1_PG_s114 >= 1
 z_k1_s1141_constraint:          var_1 + z_k1_s1141 + 1.00826 w_k1_ADM_s114
                                 + 1.06619 w_k1_AXP_s114
                                 + 1.03683 w_k1_DIS_s114
                                 + 1.08811 w_k1_FDX_s114
                                 + 0.795913 w_k1_IBM_s114
                                 + 1.373 w_k1_INTC_s114 + 0.926464 w_k1_KO_s114
                                 + 0.914188 w_k1_MAT_s114
                                 + 1.35623 w_k1_MSFT_s114
                                 + 0.981573 w_k1_PG_s114 >= 1
 z_k1_s1142_constraint:          var_1 + z_k1_s1142 + 0.899004 w_k1_ADM_s114
                                 + 0.952993 w_k1_AXP_s114
                                 + 1.03438 w_k1_DIS_s114
                                 + 0.963531 w_k1_FDX_s114
                                 + 1.16234 w_k1_IBM_s114
                                 + 1.00376 w_k1_INTC_s114 + 1.1832 w_k1_KO_s114
                                 + 1.34531 w_k1_MAT_s114
                                 + 1.15166 w_k1_MSFT_s114
                                 + 1.16526 w_k1_PG_s114 >= 1
 z_k1_s1143_constraint:          var_1 + z_k1_s1143 + 1.07607 w_k1_ADM_s114
                                 + 0.814229 w_k1_AXP_s114
                                 + 0.874043 w_k1_DIS_s114
                                 + 0.881723 w_k1_FDX_s114
                                 + 0.843428 w_k1_IBM_s114
                                 + 0.840019 w_k1_INTC_s114
                                 + 1.00332 w_k1_KO_s114
                                 + 0.950314 w_k1_MAT_s114
                                 + 1.01175 w_k1_MSFT_s114
                                 + 0.944548 w_k1_PG_s114 >= 1
 z_k1_s1144_constraint:          var_1 + z_k1_s1144 + 0.756734 w_k1_ADM_s114
                                 + 0.593267 w_k1_AXP_s114
                                 + 0.725576 w_k1_DIS_s114
                                 + 0.776638 w_k1_FDX_s114
                                 + 0.914697 w_k1_IBM_s114
                                 + 0.823534 w_k1_INTC_s114
                                 + 0.835192 w_k1_KO_s114
                                 + 0.870301 w_k1_MAT_s114
                                 + 0.923547 w_k1_MSFT_s114
                                 + 0.898443 w_k1_PG_s114 >= 1
 z_k1_s1145_constraint:          var_1 + z_k1_s1145 + 1.33783 w_k1_ADM_s114
                                 + 1.50265 w_k1_AXP_s114
                                 + 1.54111 w_k1_DIS_s114
                                 + 1.24613 w_k1_FDX_s114
                                 + 1.58199 w_k1_IBM_s114
                                 + 1.20552 w_k1_INTC_s114 + 1.6238 w_k1_KO_s114
                                 + 2.16729 w_k1_MAT_s114
                                 + 1.36616 w_k1_MSFT_s114 + 1.3267 w_k1_PG_s114
                                 >= 1
 z_k1_s1146_constraint:          var_1 + z_k1_s1146 + 1.04782 w_k1_ADM_s114
                                 + 1.01881 w_k1_AXP_s114
                                 + 1.13368 w_k1_DIS_s114
                                 + 0.900878 w_k1_FDX_s114
                                 + 1.5729 w_k1_IBM_s114
                                 + 1.03675 w_k1_INTC_s114
                                 + 1.21582 w_k1_KO_s114 + 1.42207 w_k1_MAT_s114
                                 + 1.07255 w_k1_MSFT_s114
                                 + 1.00014 w_k1_PG_s114 >= 1
 z_k1_s1147_constraint:          var_1 + z_k1_s1147 + 1.02823 w_k1_ADM_s114
                                 + 0.789002 w_k1_AXP_s114
                                 + 0.733987 w_k1_DIS_s114
                                 + 0.875339 w_k1_FDX_s114
                                 + 0.570817 w_k1_IBM_s114
                                 + 0.924834 w_k1_INTC_s114
                                 + 0.69558 w_k1_KO_s114
                                 + 0.502915 w_k1_MAT_s114
                                 + 0.96546 w_k1_MSFT_s114
                                 + 0.810476 w_k1_PG_s114 >= 1
 z_k1_s1150_constraint:          var_1 + z_k1_s1150 + 1.1133 w_k1_ADM_s115
                                 + 1.35381 w_k1_AXP_s115
                                 + 1.18104 w_k1_DIS_s115
                                 + 1.14347 w_k1_FDX_s115
                                 + 1.36948 w_k1_IBM_s115
                                 + 1.04431 w_k1_INTC_s115
                                 + 0.973689 w_k1_KO_s115
                                 + 1.02995 w_k1_MAT_s115
                                 + 1.11007 w_k1_MSFT_s115
                                 + 1.12481 w_k1_PG_s115 >= 1
 z_k1_s1151_constraint:          var_1 + z_k1_s1151 + 0.790833 w_k1_ADM_s115
                                 + 0.911555 w_k1_AXP_s115
                                 + 0.910979 w_k1_DIS_s115
                                 + 0.852614 w_k1_FDX_s115
                                 + 1.06526 w_k1_IBM_s115
                                 + 0.798113 w_k1_INTC_s115
                                 + 0.886457 w_k1_KO_s115
                                 + 1.05386 w_k1_MAT_s115
                                 + 0.934159 w_k1_MSFT_s115
                                 + 0.938544 w_k1_PG_s115 >= 1
 z_k1_s1152_constraint:          var_1 + z_k1_s1152 + 1.07901 w_k1_ADM_s115
                                 + 1.2592 w_k1_AXP_s115 + 1.15247 w_k1_DIS_s115
                                 + 1.11768 w_k1_FDX_s115 + 1.026 w_k1_IBM_s115
                                 + 1.10092 w_k1_INTC_s115
                                 + 0.967966 w_k1_KO_s115
                                 + 1.03694 w_k1_MAT_s115 + 1.072 w_k1_MSFT_s115
                                 + 1.15543 w_k1_PG_s115 >= 1
 z_k1_s1153_constraint:          var_1 + z_k1_s1153 + 0.848494 w_k1_ADM_s115
                                 + 1.0673 w_k1_AXP_s115 + 1.27073 w_k1_DIS_s115
                                 + 1.05486 w_k1_FDX_s115
                                 + 1.69258 w_k1_IBM_s115 + 1.102 w_k1_INTC_s115
                                 + 1.29572 w_k1_KO_s115
                                 + 0.991132 w_k1_MAT_s115
                                 + 0.990575 w_k1_MSFT_s115
                                 + 0.987677 w_k1_PG_s115 >= 1
 z_k1_s1154_constraint:          var_1 + z_k1_s1154 + 0.998435 w_k1_ADM_s115
                                 + 0.913482 w_k1_AXP_s115
                                 + 0.864581 w_k1_DIS_s115
                                 + 1.05561 w_k1_FDX_s115
                                 + 0.793855 w_k1_IBM_s115
                                 + 0.995784 w_k1_INTC_s115
                                 + 0.739408 w_k1_KO_s115
                                 + 0.520856 w_k1_MAT_s115
                                 + 0.970475 w_k1_MSFT_s115
                                 + 0.864091 w_k1_PG_s115 >= 1
 z_k1_s1155_constraint:          var_1 + z_k1_s1155 + 1.12552 w_k1_ADM_s115
                                 + 0.971059 w_k1_AXP_s115
                                 + 1.48177 w_k1_DIS_s115
                                 + 1.35575 w_k1_FDX_s115
                                 + 0.765854 w_k1_IBM_s115
                                 + 1.15294 w_k1_INTC_s115
                                 + 0.997964 w_k1_KO_s115
                                 + 0.828144 w_k1_MAT_s115
                                 + 1.39027 w_k1_MSFT_s115
                                 + 1.16514 w_k1_PG_s115 >= 1
 z_k1_s1156_constraint:          var_1 + z_k1_s1156 + 0.75003 w_k1_ADM_s115
                                 + 0.6358 w_k1_AXP_s115
                                 + 0.905172 w_k1_DIS_s115
                                 + 0.835992 w_k1_FDX_s115
                                 + 1.09702 w_k1_IBM_s115
                                 + 0.993866 w_k1_INTC_s115
                                 + 0.905282 w_k1_KO_s115
                                 + 0.579192 w_k1_MAT_s115
                                 + 0.982753 w_k1_MSFT_s115
                                 + 0.861848 w_k1_PG_s115 >= 1
 z_k1_s1157_constraint:          var_1 + z_k1_s1157 + 0.829316 w_k1_ADM_s115
                                 + 0.707101 w_k1_AXP_s115
                                 + 0.597416 w_k1_DIS_s115
                                 + 0.82487 w_k1_FDX_s115
                                 + 0.577037 w_k1_IBM_s115
                                 + 0.886542 w_k1_INTC_s115
                                 + 0.621063 w_k1_KO_s115
                                 + 0.469478 w_k1_MAT_s115
                                 + 0.894785 w_k1_MSFT_s115
                                 + 0.848806 w_k1_PG_s115 >= 1
 z_k1_s1160_constraint:          var_1 + z_k1_s1160 + 1.01058 w_k1_ADM_s116
                                 + 1.00747 w_k1_AXP_s116
                                 + 1.05272 w_k1_DIS_s116
                                 + 1.01679 w_k1_FDX_s116
                                 + 0.848128 w_k1_IBM_s116
                                 + 1.02798 w_k1_INTC_s116
                                 + 1.02694 w_k1_KO_s116
                                 + 0.757833 w_k1_MAT_s116
                                 + 1.12417 w_k1_MSFT_s116
                                 + 1.02041 w_k1_PG_s116 >= 1
 z_k1_s1161_constraint:          var_1 + z_k1_s1161 + 0.990686 w_k1_ADM_s116
                                 + 0.919801 w_k1_AXP_s116
                                 + 1.21006 w_k1_DIS_s116
                                 + 0.932367 w_k1_FDX_s116
                                 + 1.10033 w_k1_IBM_s116
                                 + 1.20172 w_k1_INTC_s116
                                 + 1.29469 w_k1_KO_s116
                                 + 0.793733 w_k1_MAT_s116
                                 + 1.09746 w_k1_MSFT_s116
                                 + 0.990821 w_k1_PG_s116 >= 1
 z_k1_s1162_constraint:          var_1 + z_k1_s1162 + 1.12189 w_k1_ADM_s116
                                 + 1.22823 w_k1_AXP_s116
                                 + 1.45633 w_k1_DIS_s116
                                 + 1.34084 w_k1_FDX_s116
                                 + 0.742011 w_k1_IBM_s116
                                 + 1.19218 w_k1_INTC_s116
                                 + 1.02358 w_k1_KO_s116
                                 + 0.678595 w_k1_MAT_s116
                                 + 1.30397 w_k1_MSFT_s116
                                 + 1.02144 w_k1_PG_s116 >= 1
 z_k1_s1163_constraint:          var_1 + z_k1_s1163 + 0.953492 w_k1_ADM_s116
                                 + 0.996735 w_k1_AXP_s116
                                 + 1.25106 w_k1_DIS_s116
                                 + 0.957548 w_k1_FDX_s116
                                 + 1.28201 w_k1_IBM_s116
                                 + 1.23216 w_k1_INTC_s116
                                 + 1.23528 w_k1_KO_s116
                                 + 0.927915 w_k1_MAT_s116
                                 + 1.19602 w_k1_MSFT_s116
                                 + 0.96687 w_k1_PG_s116 >= 1
 z_k1_s1164_constraint:          var_1 + z_k1_s1164 + 1.03032 w_k1_ADM_s116
                                 + 1.13126 w_k1_AXP_s116
                                 + 1.05528 w_k1_DIS_s116
                                 + 0.966285 w_k1_FDX_s116
                                 + 1.57994 w_k1_IBM_s116
                                 + 0.990962 w_k1_INTC_s116
                                 + 1.23981 w_k1_KO_s116 + 1.21521 w_k1_MAT_s116
                                 + 1.02891 w_k1_MSFT_s116
                                 + 1.11044 w_k1_PG_s116 >= 1
 z_k1_s1165_constraint:          var_1 + z_k1_s1165 + 1.18194 w_k1_ADM_s116
                                 + 1.62212 w_k1_AXP_s116
                                 + 1.30522 w_k1_DIS_s116
                                 + 1.27918 w_k1_FDX_s116
                                 + 1.15244 w_k1_IBM_s116
                                 + 1.30852 w_k1_INTC_s116
                                 + 1.38802 w_k1_KO_s116 + 1.37231 w_k1_MAT_s116
                                 + 1.25279 w_k1_MSFT_s116
                                 + 1.20341 w_k1_PG_s116 >= 1
 z_k1_s1166_constraint:          var_1 + z_k1_s1166 + 0.824907 w_k1_ADM_s116
                                 + 0.80332 w_k1_AXP_s116
                                 + 0.744438 w_k1_DIS_s116
                                 + 0.764571 w_k1_FDX_s116
                                 + 1.10656 w_k1_IBM_s116
                                 + 0.844565 w_k1_INTC_s116
                                 + 1.03871 w_k1_KO_s116
                                 + 0.940947 w_k1_MAT_s116
                                 + 0.826602 w_k1_MSFT_s116
                                 + 0.876167 w_k1_PG_s116 >= 1
 z_k1_s1167_constraint:          var_1 + z_k1_s1167 + 0.970047 w_k1_ADM_s116
                                 + 0.862935 w_k1_AXP_s116
                                 + 0.798682 w_k1_DIS_s116
                                 + 0.924112 w_k1_FDX_s116
                                 + 0.616638 w_k1_IBM_s116
                                 + 1.04377 w_k1_INTC_s116
                                 + 0.809669 w_k1_KO_s116
                                 + 0.499374 w_k1_MAT_s116
                                 + 0.93886 w_k1_MSFT_s116
                                 + 0.832442 w_k1_PG_s116 >= 1
 z_k1_s1170_constraint:          var_1 + z_k1_s1170 + 0.880596 w_k1_ADM_s117
                                 + 0.766447 w_k1_AXP_s117
                                 + 0.845868 w_k1_DIS_s117
                                 + 0.950554 w_k1_FDX_s117
                                 + 0.771063 w_k1_IBM_s117
                                 + 1.23082 w_k1_INTC_s117
                                 + 0.889946 w_k1_KO_s117
                                 + 0.817724 w_k1_MAT_s117
                                 + 0.989629 w_k1_MSFT_s117
                                 + 0.945735 w_k1_PG_s117 >= 1
 z_k1_s1171_constraint:          var_1 + z_k1_s1171 + 0.987243 w_k1_ADM_s117
                                 + 1.24977 w_k1_AXP_s117
                                 + 1.06811 w_k1_DIS_s117
                                 + 1.05743 w_k1_FDX_s117
                                 + 1.00604 w_k1_IBM_s117
                                 + 1.29369 w_k1_INTC_s117
                                 + 1.12947 w_k1_KO_s117 + 1.14327 w_k1_MAT_s117
                                 + 1.06169 w_k1_MSFT_s117
                                 + 0.972327 w_k1_PG_s117 >= 1
 z_k1_s1172_constraint:          var_1 + z_k1_s1172 + 1.45707 w_k1_ADM_s117
                                 + 1.37078 w_k1_AXP_s117
                                 + 1.45009 w_k1_DIS_s117
                                 + 1.38262 w_k1_FDX_s117
                                 + 0.904788 w_k1_IBM_s117
                                 + 1.0643 w_k1_INTC_s117 + 1.02798 w_k1_KO_s117
                                 + 1.18981 w_k1_MAT_s117
                                 + 1.16042 w_k1_MSFT_s117
                                 + 0.987434 w_k1_PG_s117 >= 1
 z_k1_s1173_constraint:          var_1 + z_k1_s1173 + 0.885638 w_k1_ADM_s117
                                 + 0.842582 w_k1_AXP_s117
                                 + 0.892708 w_k1_DIS_s117
                                 + 0.812317 w_k1_FDX_s117
                                 + 1.27072 w_k1_IBM_s117
                                 + 1.04647 w_k1_INTC_s117
                                 + 1.08959 w_k1_KO_s117 + 1.01432 w_k1_MAT_s117
                                 + 0.874569 w_k1_MSFT_s117
                                 + 0.849122 w_k1_PG_s117 >= 1
 z_k1_s1174_constraint:          var_1 + z_k1_s1174 + 1.19019 w_k1_ADM_s117
                                 + 1.02637 w_k1_AXP_s117
                                 + 1.03157 w_k1_DIS_s117
                                 + 1.17277 w_k1_FDX_s117
                                 + 0.781098 w_k1_IBM_s117
                                 + 1.03612 w_k1_INTC_s117
                                 + 1.10287 w_k1_KO_s117
                                 + 0.820187 w_k1_MAT_s117
                                 + 1.25082 w_k1_MSFT_s117
                                 + 1.09826 w_k1_PG_s117 >= 1
 z_k1_s1175_constraint:          var_1 + z_k1_s1175 + 0.720889 w_k1_ADM_s117
                                 + 0.660724 w_k1_AXP_s117
                                 + 0.638651 w_k1_DIS_s117
                                 + 0.767717 w_k1_FDX_s117
                                 + 0.743146 w_k1_IBM_s117
                                 + 0.884101 w_k1_INTC_s117
                                 + 0.801044 w_k1_KO_s117
                                 + 0.643693 w_k1_MAT_s117
                                 + 0.863995 w_k1_MSFT_s117
                                 + 0.783074 w_k1_PG_s117 >= 1
 z_k1_s1176_constraint:          var_1 + z_k1_s1176 + 1.45521 w_k1_ADM_s117
                                 + 1.17763 w_k1_AXP_s117
                                 + 1.77637 w_k1_DIS_s117
                                 + 1.27445 w_k1_FDX_s117
                                 + 1.13584 w_k1_IBM_s117
                                 + 1.06544 w_k1_INTC_s117
                                 + 1.38868 w_k1_KO_s117 + 1.38849 w_k1_MAT_s117
                                 + 1.33562 w_k1_MSFT_s117
                                 + 1.19782 w_k1_PG_s117 >= 1
 z_k1_s1177_constraint:          var_1 + z_k1_s1177 + 1.11788 w_k1_ADM_s117
                                 + 1.43486 w_k1_AXP_s117
                                 + 1.68549 w_k1_DIS_s117
                                 + 1.30825 w_k1_FDX_s117
                                 + 1.57058 w_k1_IBM_s117
                                 + 1.64363 w_k1_INTC_s117
                                 + 1.50962 w_k1_KO_s117 + 1.35128 w_k1_MAT_s117
                                 + 1.29019 w_k1_MSFT_s117
                                 + 1.04454 w_k1_PG_s117 >= 1
 k1_return_constraint:           0.0154828427847128 w_k1_ADM_s000
                                 + 0.0205641499915023 w_k1_ADM_s001
                                 + 0.020376786629819 w_k1_ADM_s002
                                 + 0.0148310486108481 w_k1_ADM_s003
                                 + 0.0235044744820184 w_k1_ADM_s004
                                 + 0.0245383067236913 w_k1_ADM_s005
                                 + 0.0105869112015914 w_k1_ADM_s006
                                 + 0.0330390434167651 w_k1_ADM_s007
                                 + 0.0340371095592651 w_k1_ADM_s010
                                 + 0.0648681906475847 w_k1_ADM_s011
                                 + 0.0746312353458862 w_k1_ADM_s012
                                 + 0.0466064394054926 w_k1_ADM_s013
                                 + 0.0333971071618846 w_k1_ADM_s014
                                 + 0.129040697956548 w_k1_ADM_s015
                                 + 0.0356277844458607 w_k1_ADM_s016
                                 + 0.0655267046485292 w_k1_ADM_s017
                                 + 0.00907253977125162 w_k1_ADM_s100
                                 + 0.0196908878106456 w_k1_ADM_s101
                                 + 0.0272164161848747 w_k1_ADM_s102
                                 + 0.0315774052850009 w_k1_ADM_s103
                                 + 0.0137050020275113 w_k1_ADM_s104
                                 + 0.012513166948051 w_k1_ADM_s105
                                 + 0.0221359631101415 w_k1_ADM_s106
                                 + 0.0361993623913124 w_k1_ADM_s107
                                 + 0.0112859547760457 w_k1_ADM_s110
                                 + 0.0123420427031113 w_k1_ADM_s111
                                 + 0.0209350844790477 w_k1_ADM_s112
                                 + 0.0151170905491103 w_k1_ADM_s113
                                 + 0.0144604649546244 w_k1_ADM_s114
                                 + 0.0293096466713302 w_k1_ADM_s115
                                 + 0.0466205454124924 w_k1_ADM_s116
                                 + 0.0281601090909274 w_k1_ADM_s117
                                 + 0.0159328835096239 w_k1_AXP_s000
                                 + 0.0229006725331348 w_k1_AXP_s001
                                 + 0.0198279413284102 w_k1_AXP_s002
                                 + 0.0165589081798614 w_k1_AXP_s003
                                 + 0.0234426176789751 w_k1_AXP_s004
                                 + 0.0244552146655346 w_k1_AXP_s005
                                 + 0.010517514034392 w_k1_AXP_s006
                                 + 0.0381484279178547 w_k1_AXP_s007
                                 + 0.033716276778311 w_k1_AXP_s010
                                 + 0.0657887884122202 w_k1_AXP_s011
                                 + 0.0759019272430466 w_k1_AXP_s012
                                 + 0.0449357565512364 w_k1_AXP_s013
                                 + 0.038925724505177 w_k1_AXP_s014
                                 + 0.143436976699364 w_k1_AXP_s015
                                 + 0.0363777588749463 w_k1_AXP_s016
                                 + 0.0598963774850208 w_k1_AXP_s017
                                 + 0.0115952475888988 w_k1_AXP_s100
                                 + 0.0181903519040255 w_k1_AXP_s101
                                 + 0.0255073188721267 w_k1_AXP_s102
                                 + 0.0370785614220241 w_k1_AXP_s103
                                 + 0.0141726973323128 w_k1_AXP_s104
                                 + 0.0126546812487049 w_k1_AXP_s105
                                 + 0.0221896688584921 w_k1_AXP_s106
                                 + 0.0378151542842305 w_k1_AXP_s107
                                 + 0.0121185917540895 w_k1_AXP_s110
                                 + 0.0164844386390576 w_k1_AXP_s111
                                 + 0.0209088344933866 w_k1_AXP_s112
                                 + 0.015730034224786 w_k1_AXP_s113
                                 + 0.0138553217697157 w_k1_AXP_s114
                                 + 0.0310491783504055 w_k1_AXP_s115
                                 + 0.0490973446255769 w_k1_AXP_s116
                                 + 0.027832704877111 w_k1_AXP_s117
                                 + 0.0185657883003705 w_k1_DIS_s000
                                 + 0.0215199139147091 w_k1_DIS_s001
                                 + 0.0192321838608873 w_k1_DIS_s002
                                 + 0.0174803675912441 w_k1_DIS_s003
                                 + 0.0251391162898604 w_k1_DIS_s004
                                 + 0.0252622730417549 w_k1_DIS_s005
                                 + 0.0110878406473922 w_k1_DIS_s006
                                 + 0.0365360344598549 w_k1_DIS_s007
                                 + 0.0359833706079072 w_k1_DIS_s010
                                 + 0.0619367982614508 w_k1_DIS_s011
                                 + 0.0745671110508644 w_k1_DIS_s012
                                 + 0.0479745776915579 w_k1_DIS_s013
                                 + 0.0380490854600793 w_k1_DIS_s014
                                 + 0.141254654495315 w_k1_DIS_s015
                                 + 0.0350035527941069 w_k1_DIS_s016
                                 + 0.0677296074675371 w_k1_DIS_s017
                                 + 0.011019204712101 w_k1_DIS_s100
                                 + 0.0190937330186501 w_k1_DIS_s101
                                 + 0.0284959815886447 w_k1_DIS_s102
                                 + 0.0351950249229021 w_k1_DIS_s103
                                 + 0.0147754517445029 w_k1_DIS_s104
                                 + 0.0121835188535575 w_k1_DIS_s105
                                 + 0.0202199739971912 w_k1_DIS_s106
                                 + 0.0350804246034421 w_k1_DIS_s107
                                 + 0.0125689238927368 w_k1_DIS_s110
                                 + 0.0159929833369218 w_k1_DIS_s111
                                 + 0.021646682345789 w_k1_DIS_s112
                                 + 0.0148008681052979 w_k1_DIS_s113
                                 + 0.0144218287156663 w_k1_DIS_s114
                                 + 0.0317562073045474 w_k1_DIS_s115
                                 + 0.0495786181718703 w_k1_DIS_s116
                                 + 0.0295210376651506 w_k1_DIS_s117
                                 + 0.0174106755218036 w_k1_FDX_s000
                                 + 0.0215232078028901 w_k1_FDX_s001
                                 + 0.0191968036297982 w_k1_FDX_s002
                                 + 0.0171090095410362 w_k1_FDX_s003
                                 + 0.023285359407042 w_k1_FDX_s004
                                 + 0.0238108884765963 w_k1_FDX_s005
                                 + 0.010906354900282 w_k1_FDX_s006
                                 + 0.0348415621382547 w_k1_FDX_s007
                                 + 0.0340963395352692 w_k1_FDX_s010
                                 + 0.0658351053953511 w_k1_FDX_s011
                                 + 0.0680052780185218 w_k1_FDX_s012
                                 + 0.0464681815936135 w_k1_FDX_s013
                                 + 0.0381265519967392 w_k1_FDX_s014
                                 + 0.141311168986418 w_k1_FDX_s015
                                 + 0.0350696280535084 w_k1_FDX_s016
                                 + 0.0644957715191992 w_k1_FDX_s017
                                 + 0.0105310822643955 w_k1_FDX_s100
                                 + 0.0188920475564091 w_k1_FDX_s101
                                 + 0.0266214582466028 w_k1_FDX_s102
                                 + 0.0330547926036869 w_k1_FDX_s103
                                 + 0.0137407737794768 w_k1_FDX_s104
                                 + 0.0111325012954238 w_k1_FDX_s105
                                 + 0.0206288367765106 w_k1_FDX_s106
                                 + 0.0334877688778047 w_k1_FDX_s107
                                 + 0.0121408517227219 w_k1_FDX_s110
                                 + 0.0151336719294567 w_k1_FDX_s111
                                 + 0.0215540070617269 w_k1_FDX_s112
                                 + 0.0149781095036178 w_k1_FDX_s113
                                 + 0.0138609561283627 w_k1_FDX_s114
                                 + 0.0316612059991046 w_k1_FDX_s115
                                 + 0.0466501526500845 w_k1_FDX_s116
                                 + 0.0283203948771155 w_k1_FDX_s117
                                 + 0.0172754662114405 w_k1_IBM_s000
                                 + 0.0212983131688666 w_k1_IBM_s001
                                 + 0.0189382683719588 w_k1_IBM_s002
                                 + 0.01781517117847 w_k1_IBM_s003
                                 + 0.0242084252940428 w_k1_IBM_s004
                                 + 0.0251961900095572 w_k1_IBM_s005
                                 + 0.0109313525601377 w_k1_IBM_s006
                                 + 0.0356655084630801 w_k1_IBM_s007
                                 + 0.0344012818746184 w_k1_IBM_s010
                                 + 0.0668837657104664 w_k1_IBM_s011
                                 + 0.0681937876969937 w_k1_IBM_s012
                                 + 0.0449845244717216 w_k1_IBM_s013
                                 + 0.0384466241167301 w_k1_IBM_s014
                                 + 0.146788805915529 w_k1_IBM_s015
                                 + 0.0356985598772143 w_k1_IBM_s016
                                 + 0.0686949739853973 w_k1_IBM_s017
                                 + 0.0108677585871458 w_k1_IBM_s100
                                 + 0.0196748667214969 w_k1_IBM_s101
                                 + 0.0270225672589159 w_k1_IBM_s102
                                 + 0.0338039246230572 w_k1_IBM_s103
                                 + 0.0137080509767124 w_k1_IBM_s104
                                 + 0.0112628684034899 w_k1_IBM_s105
                                 + 0.0204528314540279 w_k1_IBM_s106
                                 + 0.0354535138653056 w_k1_IBM_s107
                                 + 0.0122925692061782 w_k1_IBM_s110
                                 + 0.0149354264206571 w_k1_IBM_s111
                                 + 0.0215494979826309 w_k1_IBM_s112
                                 + 0.0162987921681443 w_k1_IBM_s113
                                 + 0.0138297074147904 w_k1_IBM_s114
                                 + 0.0324579741001958 w_k1_IBM_s115
                                 + 0.0480140490723831 w_k1_IBM_s116
                                 + 0.0271431008960539 w_k1_IBM_s117
                                 + 0.0184181185280175 w_k1_INTC_s000
                                 + 0.0264543489238572 w_k1_INTC_s001
                                 + 0.0169365790945946 w_k1_INTC_s002
                                 + 0.0170507539340561 w_k1_INTC_s003
                                 + 0.02341250445972 w_k1_INTC_s004
                                 + 0.0271479907607582 w_k1_INTC_s005
                                 + 0.0104450608143921 w_k1_INTC_s006
                                 + 0.0328751970641678 w_k1_INTC_s007
                                 + 0.0384301639949224 w_k1_INTC_s010
                                 + 0.0693465706262052 w_k1_INTC_s011
                                 + 0.0686683754709007 w_k1_INTC_s012
                                 + 0.0504041129398257 w_k1_INTC_s013
                                 + 0.0367499029581001 w_k1_INTC_s014
                                 + 0.158292913226341 w_k1_INTC_s015
                                 + 0.0341012820399709 w_k1_INTC_s016
                                 + 0.0660333152131622 w_k1_INTC_s017
                                 + 0.011483172405071 w_k1_INTC_s100
                                 + 0.0180860719675979 w_k1_INTC_s101
                                 + 0.0277063663712618 w_k1_INTC_s102
                                 + 0.0370818158893809 w_k1_INTC_s103
                                 + 0.0116503410452323 w_k1_INTC_s104
                                 + 0.0124520811453282 w_k1_INTC_s105
                                 + 0.0224056459555469 w_k1_INTC_s106
                                 + 0.0313688367839794 w_k1_INTC_s107
                                 + 0.0117595982102703 w_k1_INTC_s110
                                 + 0.0169180612745837 w_k1_INTC_s111
                                 + 0.0202758467907664 w_k1_INTC_s112
                                 + 0.0156882400985208 w_k1_INTC_s113
                                 + 0.0151044739045332 w_k1_INTC_s114
                                 + 0.0305408165332256 w_k1_INTC_s115
                                 + 0.0501452897347283 w_k1_INTC_s116
                                 + 0.0307634881889669 w_k1_INTC_s117
                                 + 0.0178415554966228 w_k1_KO_s000
                                 + 0.0196562880864705 w_k1_KO_s001
                                 + 0.018447811344089 w_k1_KO_s002
                                 + 0.018119847578679 w_k1_KO_s003
                                 + 0.0247243885508436 w_k1_KO_s004
                                 + 0.0254173964374377 w_k1_KO_s005
                                 + 0.00994166166139628 w_k1_KO_s006
                                 + 0.0378748804628046 w_k1_KO_s007
                                 + 0.0364216943798318 w_k1_KO_s010
                                 + 0.0623445597178153 w_k1_KO_s011
                                 + 0.0706952872368897 w_k1_KO_s012
                                 + 0.0488786776503151 w_k1_KO_s013
                                 + 0.0375618930950317 w_k1_KO_s014
                                 + 0.141696646972026 w_k1_KO_s015
                                 + 0.0343164118491475 w_k1_KO_s016
                                 + 0.0728704411986573 w_k1_KO_s017
                                 + 0.0112315823181229 w_k1_KO_s100
                                 + 0.0195429859691431 w_k1_KO_s101
                                 + 0.0263749467116531 w_k1_KO_s102
                                 + 0.0367037916366342 w_k1_KO_s103
                                 + 0.0125725332110956 w_k1_KO_s104
                                 + 0.0114214286637042 w_k1_KO_s105
                                 + 0.0206244466691274 w_k1_KO_s106
                                 + 0.0334335377906605 w_k1_KO_s107
                                 + 0.0118775905538716 w_k1_KO_s110
                                 + 0.0141740652980031 w_k1_KO_s111
                                 + 0.0229161920584536 w_k1_KO_s112
                                 + 0.0148183372808535 w_k1_KO_s113
                                 + 0.0143777905226778 w_k1_KO_s114
                                 + 0.0280529054762061 w_k1_KO_s115
                                 + 0.051942301956904 w_k1_KO_s116
                                 + 0.0292051657273633 w_k1_KO_s117
                                 + 0.0220672271902885 w_k1_MAT_s000
                                 + 0.0216622868285124 w_k1_MAT_s001
                                 + 0.0195073507945125 w_k1_MAT_s002
                                 + 0.0193956076100277 w_k1_MAT_s003
                                 + 0.0273368518222579 w_k1_MAT_s004
                                 + 0.0225090550747967 w_k1_MAT_s005
                                 + 0.0106259001691998 w_k1_MAT_s006
                                 + 0.0407230142753655 w_k1_MAT_s007
                                 + 0.0389671397755931 w_k1_MAT_s010
                                 + 0.0624877916265091 w_k1_MAT_s011
                                 + 0.065168691164853 w_k1_MAT_s012
                                 + 0.0444715614945113 w_k1_MAT_s013
                                 + 0.0393395342825682 w_k1_MAT_s014
                                 + 0.159275094336292 w_k1_MAT_s015
                                 + 0.0366098098170091 w_k1_MAT_s016
                                 + 0.062667623168543 w_k1_MAT_s017
                                 + 0.0118757048049849 w_k1_MAT_s100
                                 + 0.0207041461831141 w_k1_MAT_s101
                                 + 0.0267437322829227 w_k1_MAT_s102
                                 + 0.0316965889480628 w_k1_MAT_s103
                                 + 0.0124982341789355 w_k1_MAT_s104
                                 + 0.0121011279684938 w_k1_MAT_s105
                                 + 0.0212550346692965 w_k1_MAT_s106
                                 + 0.0320818775511254 w_k1_MAT_s107
                                 + 0.0127206301256282 w_k1_MAT_s110
                                 + 0.0169833063212036 w_k1_MAT_s111
                                 + 0.0187209317377347 w_k1_MAT_s112
                                 + 0.0145187601567509 w_k1_MAT_s113
                                 + 0.0153126239533618 w_k1_MAT_s114
                                 + 0.0260619190073651 w_k1_MAT_s115
                                 + 0.0415437456753697 w_k1_MAT_s116
                                 + 0.0273106538761107 w_k1_MAT_s117
                                 + 0.0177009242507742 w_k1_MSFT_s000
                                 + 0.0240639565824839 w_k1_MSFT_s001
                                 + 0.0198004278100051 w_k1_MSFT_s002
                                 + 0.0172118936669262 w_k1_MSFT_s003
                                 + 0.0246110666206001 w_k1_MSFT_s004
                                 + 0.0263711598422571 w_k1_MSFT_s005
                                 + 0.0108102299743425 w_k1_MSFT_s006
                                 + 0.0342981561641163 w_k1_MSFT_s007
                                 + 0.0351646902577749 w_k1_MSFT_s010
                                 + 0.0633143419222407 w_k1_MSFT_s011
                                 + 0.0633991867025023 w_k1_MSFT_s012
                                 + 0.044885136851686 w_k1_MSFT_s013
                                 + 0.0364468821703628 w_k1_MSFT_s014
                                 + 0.13787894139658 w_k1_MSFT_s015
                                 + 0.0326731570670791 w_k1_MSFT_s016
                                 + 0.0705939067420019 w_k1_MSFT_s017
                                 + 0.011258390821746 w_k1_MSFT_s100
                                 + 0.0203355463766289 w_k1_MSFT_s101
                                 + 0.0248704624620899 w_k1_MSFT_s102
                                 + 0.0325303066121779 w_k1_MSFT_s103
                                 + 0.0123234061774792 w_k1_MSFT_s104
                                 + 0.0102008508697607 w_k1_MSFT_s105
                                 + 0.0196781659683755 w_k1_MSFT_s106
                                 + 0.0314434997812805 w_k1_MSFT_s107
                                 + 0.0132917625994687 w_k1_MSFT_s110
                                 + 0.0164100919600834 w_k1_MSFT_s111
                                 + 0.0218661449217142 w_k1_MSFT_s112
                                 + 0.0139059116175724 w_k1_MSFT_s113
                                 + 0.0161844492283991 w_k1_MSFT_s114
                                 + 0.0317506131811533 w_k1_MSFT_s115
                                 + 0.0497560773273261 w_k1_MSFT_s116
                                 + 0.0287646140323365 w_k1_MSFT_s117
                                 + 0.0176024038766105 w_k1_PG_s000
                                 + 0.0216511125531122 w_k1_PG_s001
                                 + 0.0176707134770983 w_k1_PG_s002
                                 + 0.0176636664381058 w_k1_PG_s003
                                 + 0.025248035138056 w_k1_PG_s004
                                 + 0.023503618969604 w_k1_PG_s005
                                 + 0.0113801855527896 w_k1_PG_s006
                                 + 0.0353627531347455 w_k1_PG_s007
                                 + 0.0367234547466822 w_k1_PG_s010
                                 + 0.0619084515454364 w_k1_PG_s011
                                 + 0.07421195737439 w_k1_PG_s012
                                 + 0.0466234185574997 w_k1_PG_s013
                                 + 0.0369901176284696 w_k1_PG_s014
                                 + 0.143294739267153 w_k1_PG_s015
                                 + 0.0338858215627687 w_k1_PG_s016
                                 + 0.0662864390013176 w_k1_PG_s017
                                 + 0.0110685925044039 w_k1_PG_s100
                                 + 0.0183226469604357 w_k1_PG_s101
                                 + 0.0249648977308299 w_k1_PG_s102
                                 + 0.032804083684296 w_k1_PG_s103
                                 + 0.0137178904747763 w_k1_PG_s104
                                 + 0.0102889886682919 w_k1_PG_s105
                                 + 0.0197541128743713 w_k1_PG_s106
                                 + 0.0342436734879981 w_k1_PG_s107
                                 + 0.0111871667143916 w_k1_PG_s110
                                 + 0.0143037005883049 w_k1_PG_s111
                                 + 0.0236673929205231 w_k1_PG_s112
                                 + 0.0158734873092569 w_k1_PG_s113
                                 + 0.0139143949142469 w_k1_PG_s114
                                 + 0.0305632436172037 w_k1_PG_s115
                                 + 0.0464528567442822 w_k1_PG_s116
                                 + 0.0259184407039989 w_k1_PG_s117 >= 1.09
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
