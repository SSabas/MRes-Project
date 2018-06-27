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
 w_k1_ADM_s0_return_constraint:  1.10487 w_0_ADM - w_k1_ADM_s0
                                 + 0.99 b_k1_ADM_s0 - 1.01 s_k1_ADM_s0  = 0
 w_k1_ADM_s1_return_constraint:  0.993668 w_0_ADM - w_k1_ADM_s1
                                 + 0.99 b_k1_ADM_s1 - 1.01 s_k1_ADM_s1  = 0
 w_k1_ADM_s00_return_constraint: 1.15467 w_k1_ADM_s0 - w_k1_ADM_s00
                                 + 0.99 b_k1_ADM_s00 - 1.01 s_k1_ADM_s00  = 0
 w_k1_ADM_s01_return_constraint: 0.802953 w_k1_ADM_s0 - w_k1_ADM_s01
                                 + 0.99 b_k1_ADM_s01 - 1.01 s_k1_ADM_s01  = 0
 w_k1_ADM_s10_return_constraint: 1.06743 w_k1_ADM_s1 - w_k1_ADM_s10
                                 + 0.99 b_k1_ADM_s10 - 1.01 s_k1_ADM_s10  = 0
 w_k1_ADM_s11_return_constraint: 1.17376 w_k1_ADM_s1 - w_k1_ADM_s11
                                 + 0.99 b_k1_ADM_s11 - 1.01 s_k1_ADM_s11  = 0
 w_k1_ADM_s000_return_constraint: 0.83488 w_k1_ADM_s00 - w_k1_ADM_s000
                                 + 0.99 b_k1_ADM_s000 - 1.01 s_k1_ADM_s000  = 0
 w_k1_ADM_s001_return_constraint: 1.26075 w_k1_ADM_s00 - w_k1_ADM_s001
                                 + 0.99 b_k1_ADM_s001 - 1.01 s_k1_ADM_s001  = 0
 w_k1_ADM_s002_return_constraint: 0.79591 w_k1_ADM_s00 - w_k1_ADM_s002
                                 + 0.99 b_k1_ADM_s002 - 1.01 s_k1_ADM_s002  = 0
 w_k1_ADM_s003_return_constraint: 1.02015 w_k1_ADM_s00 - w_k1_ADM_s003
                                 + 0.99 b_k1_ADM_s003 - 1.01 s_k1_ADM_s003  = 0
 w_k1_ADM_s004_return_constraint: 0.72653 w_k1_ADM_s00 - w_k1_ADM_s004
                                 + 0.99 b_k1_ADM_s004 - 1.01 s_k1_ADM_s004  = 0
 w_k1_ADM_s005_return_constraint: 1.1146 w_k1_ADM_s00 - w_k1_ADM_s005
                                 + 0.99 b_k1_ADM_s005 - 1.01 s_k1_ADM_s005  = 0
 w_k1_ADM_s006_return_constraint: 0.971312 w_k1_ADM_s00 - w_k1_ADM_s006
                                 + 0.99 b_k1_ADM_s006 - 1.01 s_k1_ADM_s006  = 0
 w_k1_ADM_s007_return_constraint: 1.33116 w_k1_ADM_s00 - w_k1_ADM_s007
                                 + 0.99 b_k1_ADM_s007 - 1.01 s_k1_ADM_s007  = 0
 w_k1_ADM_s010_return_constraint: 1.05664 w_k1_ADM_s01 - w_k1_ADM_s010
                                 + 0.99 b_k1_ADM_s010 - 1.01 s_k1_ADM_s010  = 0
 w_k1_ADM_s011_return_constraint: 0.746201 w_k1_ADM_s01 - w_k1_ADM_s011
                                 + 0.99 b_k1_ADM_s011 - 1.01 s_k1_ADM_s011  = 0
 w_k1_ADM_s012_return_constraint: 1.08448 w_k1_ADM_s01 - w_k1_ADM_s012
                                 + 0.99 b_k1_ADM_s012 - 1.01 s_k1_ADM_s012  = 0
 w_k1_ADM_s013_return_constraint: 1.20429 w_k1_ADM_s01 - w_k1_ADM_s013
                                 + 0.99 b_k1_ADM_s013 - 1.01 s_k1_ADM_s013  = 0
 w_k1_ADM_s014_return_constraint: 1.05367 w_k1_ADM_s01 - w_k1_ADM_s014
                                 + 0.99 b_k1_ADM_s014 - 1.01 s_k1_ADM_s014  = 0
 w_k1_ADM_s015_return_constraint: 1.29382 w_k1_ADM_s01 - w_k1_ADM_s015
                                 + 0.99 b_k1_ADM_s015 - 1.01 s_k1_ADM_s015  = 0
 w_k1_ADM_s016_return_constraint: 1.20817 w_k1_ADM_s01 - w_k1_ADM_s016
                                 + 0.99 b_k1_ADM_s016 - 1.01 s_k1_ADM_s016  = 0
 w_k1_ADM_s017_return_constraint: 1.01634 w_k1_ADM_s01 - w_k1_ADM_s017
                                 + 0.99 b_k1_ADM_s017 - 1.01 s_k1_ADM_s017  = 0
 w_k1_ADM_s100_return_constraint: 0.895034 w_k1_ADM_s10 - w_k1_ADM_s100
                                 + 0.99 b_k1_ADM_s100 - 1.01 s_k1_ADM_s100  = 0
 w_k1_ADM_s101_return_constraint: 1.42275 w_k1_ADM_s10 - w_k1_ADM_s101
                                 + 0.99 b_k1_ADM_s101 - 1.01 s_k1_ADM_s101  = 0
 w_k1_ADM_s102_return_constraint: 0.685982 w_k1_ADM_s10 - w_k1_ADM_s102
                                 + 0.99 b_k1_ADM_s102 - 1.01 s_k1_ADM_s102  = 0
 w_k1_ADM_s103_return_constraint: 1.04259 w_k1_ADM_s10 - w_k1_ADM_s103
                                 + 0.99 b_k1_ADM_s103 - 1.01 s_k1_ADM_s103  = 0
 w_k1_ADM_s104_return_constraint: 0.90035 w_k1_ADM_s10 - w_k1_ADM_s104
                                 + 0.99 b_k1_ADM_s104 - 1.01 s_k1_ADM_s104  = 0
 w_k1_ADM_s105_return_constraint: 1.27747 w_k1_ADM_s10 - w_k1_ADM_s105
                                 + 0.99 b_k1_ADM_s105 - 1.01 s_k1_ADM_s105  = 0
 w_k1_ADM_s106_return_constraint: 0.800371 w_k1_ADM_s10 - w_k1_ADM_s106
                                 + 0.99 b_k1_ADM_s106 - 1.01 s_k1_ADM_s106  = 0
 w_k1_ADM_s107_return_constraint: 0.75592 w_k1_ADM_s10 - w_k1_ADM_s107
                                 + 0.99 b_k1_ADM_s107 - 1.01 s_k1_ADM_s107  = 0
 w_k1_ADM_s110_return_constraint: 0.883278 w_k1_ADM_s11 - w_k1_ADM_s110
                                 + 0.99 b_k1_ADM_s110 - 1.01 s_k1_ADM_s110  = 0
 w_k1_ADM_s111_return_constraint: 0.812945 w_k1_ADM_s11 - w_k1_ADM_s111
                                 + 0.99 b_k1_ADM_s111 - 1.01 s_k1_ADM_s111  = 0
 w_k1_ADM_s112_return_constraint: 0.859976 w_k1_ADM_s11 - w_k1_ADM_s112
                                 + 0.99 b_k1_ADM_s112 - 1.01 s_k1_ADM_s112  = 0
 w_k1_ADM_s113_return_constraint: 0.756995 w_k1_ADM_s11 - w_k1_ADM_s113
                                 + 0.99 b_k1_ADM_s113 - 1.01 s_k1_ADM_s113  = 0
 w_k1_ADM_s114_return_constraint: 0.650251 w_k1_ADM_s11 - w_k1_ADM_s114
                                 + 0.99 b_k1_ADM_s114 - 1.01 s_k1_ADM_s114  = 0
 w_k1_ADM_s115_return_constraint: 1.53871 w_k1_ADM_s11 - w_k1_ADM_s115
                                 + 0.99 b_k1_ADM_s115 - 1.01 s_k1_ADM_s115  = 0
 w_k1_ADM_s116_return_constraint: 0.773021 w_k1_ADM_s11 - w_k1_ADM_s116
                                 + 0.99 b_k1_ADM_s116 - 1.01 s_k1_ADM_s116  = 0
 w_k1_ADM_s117_return_constraint: 0.711006 w_k1_ADM_s11 - w_k1_ADM_s117
                                 + 0.99 b_k1_ADM_s117 - 1.01 s_k1_ADM_s117  = 0
 w_k1_AXP_s0_return_constraint:  1.01988 w_0_AXP - w_k1_AXP_s0
                                 + 0.99 b_k1_AXP_s0 - 1.01 s_k1_AXP_s0  = 0
 w_k1_AXP_s1_return_constraint:  1.06133 w_0_AXP - w_k1_AXP_s1
                                 + 0.99 b_k1_AXP_s1 - 1.01 s_k1_AXP_s1  = 0
 w_k1_AXP_s00_return_constraint: 1.2387 w_k1_AXP_s0 - w_k1_AXP_s00
                                 + 0.99 b_k1_AXP_s00 - 1.01 s_k1_AXP_s00  = 0
 w_k1_AXP_s01_return_constraint: 0.904258 w_k1_AXP_s0 - w_k1_AXP_s01
                                 + 0.99 b_k1_AXP_s01 - 1.01 s_k1_AXP_s01  = 0
 w_k1_AXP_s10_return_constraint: 0.950806 w_k1_AXP_s1 - w_k1_AXP_s10
                                 + 0.99 b_k1_AXP_s10 - 1.01 s_k1_AXP_s10  = 0
 w_k1_AXP_s11_return_constraint: 1.07636 w_k1_AXP_s1 - w_k1_AXP_s11
                                 + 0.99 b_k1_AXP_s11 - 1.01 s_k1_AXP_s11  = 0
 w_k1_AXP_s000_return_constraint: 0.990263 w_k1_AXP_s00 - w_k1_AXP_s000
                                 + 0.99 b_k1_AXP_s000 - 1.01 s_k1_AXP_s000  = 0
 w_k1_AXP_s001_return_constraint: 1.24231 w_k1_AXP_s00 - w_k1_AXP_s001
                                 + 0.99 b_k1_AXP_s001 - 1.01 s_k1_AXP_s001  = 0
 w_k1_AXP_s002_return_constraint: 0.890171 w_k1_AXP_s00 - w_k1_AXP_s002
                                 + 0.99 b_k1_AXP_s002 - 1.01 s_k1_AXP_s002  = 0
 w_k1_AXP_s003_return_constraint: 0.886054 w_k1_AXP_s00 - w_k1_AXP_s003
                                 + 0.99 b_k1_AXP_s003 - 1.01 s_k1_AXP_s003  = 0
 w_k1_AXP_s004_return_constraint: 0.689046 w_k1_AXP_s00 - w_k1_AXP_s004
                                 + 0.99 b_k1_AXP_s004 - 1.01 s_k1_AXP_s004  = 0
 w_k1_AXP_s005_return_constraint: 1.02725 w_k1_AXP_s00 - w_k1_AXP_s005
                                 + 0.99 b_k1_AXP_s005 - 1.01 s_k1_AXP_s005  = 0
 w_k1_AXP_s006_return_constraint: 1.2767 w_k1_AXP_s00 - w_k1_AXP_s006
                                 + 0.99 b_k1_AXP_s006 - 1.01 s_k1_AXP_s006  = 0
 w_k1_AXP_s007_return_constraint: 1.76356 w_k1_AXP_s00 - w_k1_AXP_s007
                                 + 0.99 b_k1_AXP_s007 - 1.01 s_k1_AXP_s007  = 0
 w_k1_AXP_s010_return_constraint: 0.88575 w_k1_AXP_s01 - w_k1_AXP_s010
                                 + 0.99 b_k1_AXP_s010 - 1.01 s_k1_AXP_s010  = 0
 w_k1_AXP_s011_return_constraint: 0.619102 w_k1_AXP_s01 - w_k1_AXP_s011
                                 + 0.99 b_k1_AXP_s011 - 1.01 s_k1_AXP_s011  = 0
 w_k1_AXP_s012_return_constraint: 1.66234 w_k1_AXP_s01 - w_k1_AXP_s012
                                 + 0.99 b_k1_AXP_s012 - 1.01 s_k1_AXP_s012  = 0
 w_k1_AXP_s013_return_constraint: 1.24598 w_k1_AXP_s01 - w_k1_AXP_s013
                                 + 0.99 b_k1_AXP_s013 - 1.01 s_k1_AXP_s013  = 0
 w_k1_AXP_s014_return_constraint: 1.23231 w_k1_AXP_s01 - w_k1_AXP_s014
                                 + 0.99 b_k1_AXP_s014 - 1.01 s_k1_AXP_s014  = 0
 w_k1_AXP_s015_return_constraint: 1.60897 w_k1_AXP_s01 - w_k1_AXP_s015
                                 + 0.99 b_k1_AXP_s015 - 1.01 s_k1_AXP_s015  = 0
 w_k1_AXP_s016_return_constraint: 1.10862 w_k1_AXP_s01 - w_k1_AXP_s016
                                 + 0.99 b_k1_AXP_s016 - 1.01 s_k1_AXP_s016  = 0
 w_k1_AXP_s017_return_constraint: 1.00487 w_k1_AXP_s01 - w_k1_AXP_s017
                                 + 0.99 b_k1_AXP_s017 - 1.01 s_k1_AXP_s017  = 0
 w_k1_AXP_s100_return_constraint: 1.1324 w_k1_AXP_s10 - w_k1_AXP_s100
                                 + 0.99 b_k1_AXP_s100 - 1.01 s_k1_AXP_s100  = 0
 w_k1_AXP_s101_return_constraint: 1.85909 w_k1_AXP_s10 - w_k1_AXP_s101
                                 + 0.99 b_k1_AXP_s101 - 1.01 s_k1_AXP_s101  = 0
 w_k1_AXP_s102_return_constraint: 0.502829 w_k1_AXP_s10 - w_k1_AXP_s102
                                 + 0.99 b_k1_AXP_s102 - 1.01 s_k1_AXP_s102  = 0
 w_k1_AXP_s103_return_constraint: 1.03586 w_k1_AXP_s10 - w_k1_AXP_s103
                                 + 0.99 b_k1_AXP_s103 - 1.01 s_k1_AXP_s103  = 0
 w_k1_AXP_s104_return_constraint: 1.00159 w_k1_AXP_s10 - w_k1_AXP_s104
                                 + 0.99 b_k1_AXP_s104 - 1.01 s_k1_AXP_s104  = 0
 w_k1_AXP_s105_return_constraint: 1.66409 w_k1_AXP_s10 - w_k1_AXP_s105
                                 + 0.99 b_k1_AXP_s105 - 1.01 s_k1_AXP_s105  = 0
 w_k1_AXP_s106_return_constraint: 0.797173 w_k1_AXP_s10 - w_k1_AXP_s106
                                 + 0.99 b_k1_AXP_s106 - 1.01 s_k1_AXP_s106  = 0
 w_k1_AXP_s107_return_constraint: 0.719631 w_k1_AXP_s10 - w_k1_AXP_s107
                                 + 0.99 b_k1_AXP_s107 - 1.01 s_k1_AXP_s107  = 0
 w_k1_AXP_s110_return_constraint: 1.03294 w_k1_AXP_s11 - w_k1_AXP_s110
                                 + 0.99 b_k1_AXP_s110 - 1.01 s_k1_AXP_s110  = 0
 w_k1_AXP_s111_return_constraint: 1.10479 w_k1_AXP_s11 - w_k1_AXP_s111
                                 + 0.99 b_k1_AXP_s111 - 1.01 s_k1_AXP_s111  = 0
 w_k1_AXP_s112_return_constraint: 0.954873 w_k1_AXP_s11 - w_k1_AXP_s112
                                 + 0.99 b_k1_AXP_s112 - 1.01 s_k1_AXP_s112  = 0
 w_k1_AXP_s113_return_constraint: 0.815418 w_k1_AXP_s11 - w_k1_AXP_s113
                                 + 0.99 b_k1_AXP_s113 - 1.01 s_k1_AXP_s113  = 0
 w_k1_AXP_s114_return_constraint: 0.636983 w_k1_AXP_s11 - w_k1_AXP_s114
                                 + 0.99 b_k1_AXP_s114 - 1.01 s_k1_AXP_s114  = 0
 w_k1_AXP_s115_return_constraint: 1.52765 w_k1_AXP_s11 - w_k1_AXP_s115
                                 + 0.99 b_k1_AXP_s115 - 1.01 s_k1_AXP_s115  = 0
 w_k1_AXP_s116_return_constraint: 0.861285 w_k1_AXP_s11 - w_k1_AXP_s116
                                 + 0.99 b_k1_AXP_s116 - 1.01 s_k1_AXP_s116  = 0
 w_k1_AXP_s117_return_constraint: 0.656713 w_k1_AXP_s11 - w_k1_AXP_s117
                                 + 0.99 b_k1_AXP_s117 - 1.01 s_k1_AXP_s117  = 0
 w_k1_DIS_s0_return_constraint:  1.0866 w_0_DIS - w_k1_DIS_s0
                                 + 0.99 b_k1_DIS_s0 - 1.01 s_k1_DIS_s0  = 0
 w_k1_DIS_s1_return_constraint:  1.00995 w_0_DIS - w_k1_DIS_s1
                                 + 0.99 b_k1_DIS_s1 - 1.01 s_k1_DIS_s1  = 0
 w_k1_DIS_s00_return_constraint: 1.38222 w_k1_DIS_s0 - w_k1_DIS_s00
                                 + 0.99 b_k1_DIS_s00 - 1.01 s_k1_DIS_s00  = 0
 w_k1_DIS_s01_return_constraint: 0.794734 w_k1_DIS_s0 - w_k1_DIS_s01
                                 + 0.99 b_k1_DIS_s01 - 1.01 s_k1_DIS_s01  = 0
 w_k1_DIS_s10_return_constraint: 0.917795 w_k1_DIS_s1 - w_k1_DIS_s10
                                 + 0.99 b_k1_DIS_s10 - 1.01 s_k1_DIS_s10  = 0
 w_k1_DIS_s11_return_constraint: 1.17245 w_k1_DIS_s1 - w_k1_DIS_s11
                                 + 0.99 b_k1_DIS_s11 - 1.01 s_k1_DIS_s11  = 0
 w_k1_DIS_s000_return_constraint: 0.903177 w_k1_DIS_s00 - w_k1_DIS_s000
                                 + 0.99 b_k1_DIS_s000 - 1.01 s_k1_DIS_s000  = 0
 w_k1_DIS_s001_return_constraint: 1.09971 w_k1_DIS_s00 - w_k1_DIS_s001
                                 + 0.99 b_k1_DIS_s001 - 1.01 s_k1_DIS_s001  = 0
 w_k1_DIS_s002_return_constraint: 0.815667 w_k1_DIS_s00 - w_k1_DIS_s002
                                 + 0.99 b_k1_DIS_s002 - 1.01 s_k1_DIS_s002  = 0
 w_k1_DIS_s003_return_constraint: 0.97821 w_k1_DIS_s00 - w_k1_DIS_s003
                                 + 0.99 b_k1_DIS_s003 - 1.01 s_k1_DIS_s003  = 0
 w_k1_DIS_s004_return_constraint: 0.644888 w_k1_DIS_s00 - w_k1_DIS_s004
                                 + 0.99 b_k1_DIS_s004 - 1.01 s_k1_DIS_s004  = 0
 w_k1_DIS_s005_return_constraint: 1.19051 w_k1_DIS_s00 - w_k1_DIS_s005
                                 + 0.99 b_k1_DIS_s005 - 1.01 s_k1_DIS_s005  = 0
 w_k1_DIS_s006_return_constraint: 1.23361 w_k1_DIS_s00 - w_k1_DIS_s006
                                 + 0.99 b_k1_DIS_s006 - 1.01 s_k1_DIS_s006  = 0
 w_k1_DIS_s007_return_constraint: 1.82947 w_k1_DIS_s00 - w_k1_DIS_s007
                                 + 0.99 b_k1_DIS_s007 - 1.01 s_k1_DIS_s007  = 0
 w_k1_DIS_s010_return_constraint: 0.845887 w_k1_DIS_s01 - w_k1_DIS_s010
                                 + 0.99 b_k1_DIS_s010 - 1.01 s_k1_DIS_s010  = 0
 w_k1_DIS_s011_return_constraint: 0.610822 w_k1_DIS_s01 - w_k1_DIS_s011
                                 + 0.99 b_k1_DIS_s011 - 1.01 s_k1_DIS_s011  = 0
 w_k1_DIS_s012_return_constraint: 1.34466 w_k1_DIS_s01 - w_k1_DIS_s012
                                 + 0.99 b_k1_DIS_s012 - 1.01 s_k1_DIS_s012  = 0
 w_k1_DIS_s013_return_constraint: 1.38724 w_k1_DIS_s01 - w_k1_DIS_s013
                                 + 0.99 b_k1_DIS_s013 - 1.01 s_k1_DIS_s013  = 0
 w_k1_DIS_s014_return_constraint: 1.46146 w_k1_DIS_s01 - w_k1_DIS_s014
                                 + 0.99 b_k1_DIS_s014 - 1.01 s_k1_DIS_s014  = 0
 w_k1_DIS_s015_return_constraint: 2.06596 w_k1_DIS_s01 - w_k1_DIS_s015
                                 + 0.99 b_k1_DIS_s015 - 1.01 s_k1_DIS_s015  = 0
 w_k1_DIS_s016_return_constraint: 1.16415 w_k1_DIS_s01 - w_k1_DIS_s016
                                 + 0.99 b_k1_DIS_s016 - 1.01 s_k1_DIS_s016  = 0
 w_k1_DIS_s017_return_constraint: 0.980743 w_k1_DIS_s01 - w_k1_DIS_s017
                                 + 0.99 b_k1_DIS_s017 - 1.01 s_k1_DIS_s017  = 0
 w_k1_DIS_s100_return_constraint: 1.10936 w_k1_DIS_s10 - w_k1_DIS_s100
                                 + 0.99 b_k1_DIS_s100 - 1.01 s_k1_DIS_s100  = 0
 w_k1_DIS_s101_return_constraint: 2.01409 w_k1_DIS_s10 - w_k1_DIS_s101
                                 + 0.99 b_k1_DIS_s101 - 1.01 s_k1_DIS_s101  = 0
 w_k1_DIS_s102_return_constraint: 0.492969 w_k1_DIS_s10 - w_k1_DIS_s102
                                 + 0.99 b_k1_DIS_s102 - 1.01 s_k1_DIS_s102  = 0
 w_k1_DIS_s103_return_constraint: 1.12137 w_k1_DIS_s10 - w_k1_DIS_s103
                                 + 0.99 b_k1_DIS_s103 - 1.01 s_k1_DIS_s103  = 0
 w_k1_DIS_s104_return_constraint: 1.06089 w_k1_DIS_s10 - w_k1_DIS_s104
                                 + 0.99 b_k1_DIS_s104 - 1.01 s_k1_DIS_s104  = 0
 w_k1_DIS_s105_return_constraint: 1.80666 w_k1_DIS_s10 - w_k1_DIS_s105
                                 + 0.99 b_k1_DIS_s105 - 1.01 s_k1_DIS_s105  = 0
 w_k1_DIS_s106_return_constraint: 0.579112 w_k1_DIS_s10 - w_k1_DIS_s106
                                 + 0.99 b_k1_DIS_s106 - 1.01 s_k1_DIS_s106  = 0
 w_k1_DIS_s107_return_constraint: 0.643693 w_k1_DIS_s10 - w_k1_DIS_s107
                                 + 0.99 b_k1_DIS_s107 - 1.01 s_k1_DIS_s107  = 0
 w_k1_DIS_s110_return_constraint: 1.00073 w_k1_DIS_s11 - w_k1_DIS_s110
                                 + 0.99 b_k1_DIS_s110 - 1.01 s_k1_DIS_s110  = 0
 w_k1_DIS_s111_return_constraint: 0.87645 w_k1_DIS_s11 - w_k1_DIS_s111
                                 + 0.99 b_k1_DIS_s111 - 1.01 s_k1_DIS_s111  = 0
 w_k1_DIS_s112_return_constraint: 0.911369 w_k1_DIS_s11 - w_k1_DIS_s112
                                 + 0.99 b_k1_DIS_s112 - 1.01 s_k1_DIS_s112  = 0
 w_k1_DIS_s113_return_constraint: 0.658179 w_k1_DIS_s11 - w_k1_DIS_s113
                                 + 0.99 b_k1_DIS_s113 - 1.01 s_k1_DIS_s113  = 0
 w_k1_DIS_s114_return_constraint: 0.519072 w_k1_DIS_s11 - w_k1_DIS_s114
                                 + 0.99 b_k1_DIS_s114 - 1.01 s_k1_DIS_s114  = 0
 w_k1_DIS_s115_return_constraint: 1.82383 w_k1_DIS_s11 - w_k1_DIS_s115
                                 + 0.99 b_k1_DIS_s115 - 1.01 s_k1_DIS_s115  = 0
 w_k1_DIS_s116_return_constraint: 1.12626 w_k1_DIS_s11 - w_k1_DIS_s116
                                 + 0.99 b_k1_DIS_s116 - 1.01 s_k1_DIS_s116  = 0
 w_k1_DIS_s117_return_constraint: 0.651695 w_k1_DIS_s11 - w_k1_DIS_s117
                                 + 0.99 b_k1_DIS_s117 - 1.01 s_k1_DIS_s117  = 0
 w_k1_FDX_s0_return_constraint:  1.08528 w_0_FDX - w_k1_FDX_s0
                                 + 0.99 b_k1_FDX_s0 - 1.01 s_k1_FDX_s0  = 0
 w_k1_FDX_s1_return_constraint:  0.958794 w_0_FDX - w_k1_FDX_s1
                                 + 0.99 b_k1_FDX_s1 - 1.01 s_k1_FDX_s1  = 0
 w_k1_FDX_s00_return_constraint: 1.26924 w_k1_FDX_s0 - w_k1_FDX_s00
                                 + 0.99 b_k1_FDX_s00 - 1.01 s_k1_FDX_s00  = 0
 w_k1_FDX_s01_return_constraint: 0.839573 w_k1_FDX_s0 - w_k1_FDX_s01
                                 + 0.99 b_k1_FDX_s01 - 1.01 s_k1_FDX_s01  = 0
 w_k1_FDX_s10_return_constraint: 0.994407 w_k1_FDX_s1 - w_k1_FDX_s10
                                 + 0.99 b_k1_FDX_s10 - 1.01 s_k1_FDX_s10  = 0
 w_k1_FDX_s11_return_constraint: 1.06853 w_k1_FDX_s1 - w_k1_FDX_s11
                                 + 0.99 b_k1_FDX_s11 - 1.01 s_k1_FDX_s11  = 0
 w_k1_FDX_s000_return_constraint: 0.862673 w_k1_FDX_s00 - w_k1_FDX_s000
                                 + 0.99 b_k1_FDX_s000 - 1.01 s_k1_FDX_s000  = 0
 w_k1_FDX_s001_return_constraint: 1.13344 w_k1_FDX_s00 - w_k1_FDX_s001
                                 + 0.99 b_k1_FDX_s001 - 1.01 s_k1_FDX_s001  = 0
 w_k1_FDX_s002_return_constraint: 0.848091 w_k1_FDX_s00 - w_k1_FDX_s002
                                 + 0.99 b_k1_FDX_s002 - 1.01 s_k1_FDX_s002  = 0
 w_k1_FDX_s003_return_constraint: 1.05581 w_k1_FDX_s00 - w_k1_FDX_s003
                                 + 0.99 b_k1_FDX_s003 - 1.01 s_k1_FDX_s003  = 0
 w_k1_FDX_s004_return_constraint: 0.772106 w_k1_FDX_s00 - w_k1_FDX_s004
                                 + 0.99 b_k1_FDX_s004 - 1.01 s_k1_FDX_s004  = 0
 w_k1_FDX_s005_return_constraint: 1.05606 w_k1_FDX_s00 - w_k1_FDX_s005
                                 + 0.99 b_k1_FDX_s005 - 1.01 s_k1_FDX_s005  = 0
 w_k1_FDX_s006_return_constraint: 1.03147 w_k1_FDX_s00 - w_k1_FDX_s006
                                 + 0.99 b_k1_FDX_s006 - 1.01 s_k1_FDX_s006  = 0
 w_k1_FDX_s007_return_constraint: 1.42142 w_k1_FDX_s00 - w_k1_FDX_s007
                                 + 0.99 b_k1_FDX_s007 - 1.01 s_k1_FDX_s007  = 0
 w_k1_FDX_s010_return_constraint: 0.99372 w_k1_FDX_s01 - w_k1_FDX_s010
                                 + 0.99 b_k1_FDX_s010 - 1.01 s_k1_FDX_s010  = 0
 w_k1_FDX_s011_return_constraint: 0.730689 w_k1_FDX_s01 - w_k1_FDX_s011
                                 + 0.99 b_k1_FDX_s011 - 1.01 s_k1_FDX_s011  = 0
 w_k1_FDX_s012_return_constraint: 1.3735 w_k1_FDX_s01 - w_k1_FDX_s012
                                 + 0.99 b_k1_FDX_s012 - 1.01 s_k1_FDX_s012  = 0
 w_k1_FDX_s013_return_constraint: 1.33033 w_k1_FDX_s01 - w_k1_FDX_s013
                                 + 0.99 b_k1_FDX_s013 - 1.01 s_k1_FDX_s013  = 0
 w_k1_FDX_s014_return_constraint: 1.12463 w_k1_FDX_s01 - w_k1_FDX_s014
                                 + 0.99 b_k1_FDX_s014 - 1.01 s_k1_FDX_s014  = 0
 w_k1_FDX_s015_return_constraint: 1.50211 w_k1_FDX_s01 - w_k1_FDX_s015
                                 + 0.99 b_k1_FDX_s015 - 1.01 s_k1_FDX_s015  = 0
 w_k1_FDX_s016_return_constraint: 1.13609 w_k1_FDX_s01 - w_k1_FDX_s016
                                 + 0.99 b_k1_FDX_s016 - 1.01 s_k1_FDX_s016  = 0
 w_k1_FDX_s017_return_constraint: 0.883723 w_k1_FDX_s01 - w_k1_FDX_s017
                                 + 0.99 b_k1_FDX_s017 - 1.01 s_k1_FDX_s017  = 0
 w_k1_FDX_s100_return_constraint: 0.985538 w_k1_FDX_s10 - w_k1_FDX_s100
                                 + 0.99 b_k1_FDX_s100 - 1.01 s_k1_FDX_s100  = 0
 w_k1_FDX_s101_return_constraint: 1.49974 w_k1_FDX_s10 - w_k1_FDX_s101
                                 + 0.99 b_k1_FDX_s101 - 1.01 s_k1_FDX_s101  = 0
 w_k1_FDX_s102_return_constraint: 0.657589 w_k1_FDX_s10 - w_k1_FDX_s102
                                 + 0.99 b_k1_FDX_s102 - 1.01 s_k1_FDX_s102  = 0
 w_k1_FDX_s103_return_constraint: 1.14188 w_k1_FDX_s10 - w_k1_FDX_s103
                                 + 0.99 b_k1_FDX_s103 - 1.01 s_k1_FDX_s103  = 0
 w_k1_FDX_s104_return_constraint: 1.03822 w_k1_FDX_s10 - w_k1_FDX_s104
                                 + 0.99 b_k1_FDX_s104 - 1.01 s_k1_FDX_s104  = 0
 w_k1_FDX_s105_return_constraint: 1.56997 w_k1_FDX_s10 - w_k1_FDX_s105
                                 + 0.99 b_k1_FDX_s105 - 1.01 s_k1_FDX_s105  = 0
 w_k1_FDX_s106_return_constraint: 0.817042 w_k1_FDX_s10 - w_k1_FDX_s106
                                 + 0.99 b_k1_FDX_s106 - 1.01 s_k1_FDX_s106  = 0
 w_k1_FDX_s107_return_constraint: 0.709693 w_k1_FDX_s10 - w_k1_FDX_s107
                                 + 0.99 b_k1_FDX_s107 - 1.01 s_k1_FDX_s107  = 0
 w_k1_FDX_s110_return_constraint: 1.08127 w_k1_FDX_s11 - w_k1_FDX_s110
                                 + 0.99 b_k1_FDX_s110 - 1.01 s_k1_FDX_s110  = 0
 w_k1_FDX_s111_return_constraint: 0.976254 w_k1_FDX_s11 - w_k1_FDX_s111
                                 + 0.99 b_k1_FDX_s111 - 1.01 s_k1_FDX_s111  = 0
 w_k1_FDX_s112_return_constraint: 0.876637 w_k1_FDX_s11 - w_k1_FDX_s112
                                 + 0.99 b_k1_FDX_s112 - 1.01 s_k1_FDX_s112  = 0
 w_k1_FDX_s113_return_constraint: 0.764115 w_k1_FDX_s11 - w_k1_FDX_s113
                                 + 0.99 b_k1_FDX_s113 - 1.01 s_k1_FDX_s113  = 0
 w_k1_FDX_s114_return_constraint: 0.700359 w_k1_FDX_s11 - w_k1_FDX_s114
                                 + 0.99 b_k1_FDX_s114 - 1.01 s_k1_FDX_s114  = 0
 w_k1_FDX_s115_return_constraint: 1.4506 w_k1_FDX_s11 - w_k1_FDX_s115
                                 + 0.99 b_k1_FDX_s115 - 1.01 s_k1_FDX_s115  = 0
 w_k1_FDX_s116_return_constraint: 0.973339 w_k1_FDX_s11 - w_k1_FDX_s116
                                 + 0.99 b_k1_FDX_s116 - 1.01 s_k1_FDX_s116  = 0
 w_k1_FDX_s117_return_constraint: 0.691082 w_k1_FDX_s11 - w_k1_FDX_s117
                                 + 0.99 b_k1_FDX_s117 - 1.01 s_k1_FDX_s117  = 0
 w_k1_IBM_s0_return_constraint:  0.871856 w_0_IBM - w_k1_IBM_s0
                                 + 0.99 b_k1_IBM_s0 - 1.01 s_k1_IBM_s0  = 0
 w_k1_IBM_s1_return_constraint:  1.15994 w_0_IBM - w_k1_IBM_s1
                                 + 0.99 b_k1_IBM_s1 - 1.01 s_k1_IBM_s1  = 0
 w_k1_IBM_s00_return_constraint: 1.10445 w_k1_IBM_s0 - w_k1_IBM_s00
                                 + 0.99 b_k1_IBM_s00 - 1.01 s_k1_IBM_s00  = 0
 w_k1_IBM_s01_return_constraint: 0.998113 w_k1_IBM_s0 - w_k1_IBM_s01
                                 + 0.99 b_k1_IBM_s01 - 1.01 s_k1_IBM_s01  = 0
 w_k1_IBM_s10_return_constraint: 0.851917 w_k1_IBM_s1 - w_k1_IBM_s10
                                 + 0.99 b_k1_IBM_s10 - 1.01 s_k1_IBM_s10  = 0
 w_k1_IBM_s11_return_constraint: 1.36099 w_k1_IBM_s1 - w_k1_IBM_s11
                                 + 0.99 b_k1_IBM_s11 - 1.01 s_k1_IBM_s11  = 0
 w_k1_IBM_s000_return_constraint: 1.28474 w_k1_IBM_s00 - w_k1_IBM_s000
                                 + 0.99 b_k1_IBM_s000 - 1.01 s_k1_IBM_s000  = 0
 w_k1_IBM_s001_return_constraint: 0.842523 w_k1_IBM_s00 - w_k1_IBM_s001
                                 + 0.99 b_k1_IBM_s001 - 1.01 s_k1_IBM_s001  = 0
 w_k1_IBM_s002_return_constraint: 1.0152 w_k1_IBM_s00 - w_k1_IBM_s002
                                 + 0.99 b_k1_IBM_s002 - 1.01 s_k1_IBM_s002  = 0
 w_k1_IBM_s003_return_constraint: 0.678607 w_k1_IBM_s00 - w_k1_IBM_s003
                                 + 0.99 b_k1_IBM_s003 - 1.01 s_k1_IBM_s003  = 0
 w_k1_IBM_s004_return_constraint: 0.673977 w_k1_IBM_s00 - w_k1_IBM_s004
                                 + 0.99 b_k1_IBM_s004 - 1.01 s_k1_IBM_s004  = 0
 w_k1_IBM_s005_return_constraint: 1.16683 w_k1_IBM_s00 - w_k1_IBM_s005
                                 + 0.99 b_k1_IBM_s005 - 1.01 s_k1_IBM_s005  = 0
 w_k1_IBM_s006_return_constraint: 1.68489 w_k1_IBM_s00 - w_k1_IBM_s006
                                 + 0.99 b_k1_IBM_s006 - 1.01 s_k1_IBM_s006  = 0
 w_k1_IBM_s007_return_constraint: 1.50169 w_k1_IBM_s00 - w_k1_IBM_s007
                                 + 0.99 b_k1_IBM_s007 - 1.01 s_k1_IBM_s007  = 0
 w_k1_IBM_s010_return_constraint: 0.729019 w_k1_IBM_s01 - w_k1_IBM_s010
                                 + 0.99 b_k1_IBM_s010 - 1.01 s_k1_IBM_s010  = 0
 w_k1_IBM_s011_return_constraint: 0.761631 w_k1_IBM_s01 - w_k1_IBM_s011
                                 + 0.99 b_k1_IBM_s011 - 1.01 s_k1_IBM_s011  = 0
 w_k1_IBM_s012_return_constraint: 1.19828 w_k1_IBM_s01 - w_k1_IBM_s012
                                 + 0.99 b_k1_IBM_s012 - 1.01 s_k1_IBM_s012  = 0
 w_k1_IBM_s013_return_constraint: 0.770901 w_k1_IBM_s01 - w_k1_IBM_s013
                                 + 0.99 b_k1_IBM_s013 - 1.01 s_k1_IBM_s013  = 0
 w_k1_IBM_s014_return_constraint: 1.56196 w_k1_IBM_s01 - w_k1_IBM_s014
                                 + 0.99 b_k1_IBM_s014 - 1.01 s_k1_IBM_s014  = 0
 w_k1_IBM_s015_return_constraint: 1.85464 w_k1_IBM_s01 - w_k1_IBM_s015
                                 + 0.99 b_k1_IBM_s015 - 1.01 s_k1_IBM_s015  = 0
 w_k1_IBM_s016_return_constraint: 1.01075 w_k1_IBM_s01 - w_k1_IBM_s016
                                 + 0.99 b_k1_IBM_s016 - 1.01 s_k1_IBM_s016  = 0
 w_k1_IBM_s017_return_constraint: 1.34725 w_k1_IBM_s01 - w_k1_IBM_s017
                                 + 0.99 b_k1_IBM_s017 - 1.01 s_k1_IBM_s017  = 0
 w_k1_IBM_s100_return_constraint: 1.39292 w_k1_IBM_s10 - w_k1_IBM_s100
                                 + 0.99 b_k1_IBM_s100 - 1.01 s_k1_IBM_s100  = 0
 w_k1_IBM_s101_return_constraint: 1.58385 w_k1_IBM_s10 - w_k1_IBM_s101
                                 + 0.99 b_k1_IBM_s101 - 1.01 s_k1_IBM_s101  = 0
 w_k1_IBM_s102_return_constraint: 0.497548 w_k1_IBM_s10 - w_k1_IBM_s102
                                 + 0.99 b_k1_IBM_s102 - 1.01 s_k1_IBM_s102  = 0
 w_k1_IBM_s103_return_constraint: 0.734115 w_k1_IBM_s10 - w_k1_IBM_s103
                                 + 0.99 b_k1_IBM_s103 - 1.01 s_k1_IBM_s103  = 0
 w_k1_IBM_s104_return_constraint: 1.0284 w_k1_IBM_s10 - w_k1_IBM_s104
                                 + 0.99 b_k1_IBM_s104 - 1.01 s_k1_IBM_s104  = 0
 w_k1_IBM_s105_return_constraint: 1.13686 w_k1_IBM_s10 - w_k1_IBM_s105
                                 + 0.99 b_k1_IBM_s105 - 1.01 s_k1_IBM_s105  = 0
 w_k1_IBM_s106_return_constraint: 0.635494 w_k1_IBM_s10 - w_k1_IBM_s106
                                 + 0.99 b_k1_IBM_s106 - 1.01 s_k1_IBM_s106  = 0
 w_k1_IBM_s107_return_constraint: 0.873695 w_k1_IBM_s10 - w_k1_IBM_s107
                                 + 0.99 b_k1_IBM_s107 - 1.01 s_k1_IBM_s107  = 0
 w_k1_IBM_s110_return_constraint: 0.854608 w_k1_IBM_s11 - w_k1_IBM_s110
                                 + 0.99 b_k1_IBM_s110 - 1.01 s_k1_IBM_s110  = 0
 w_k1_IBM_s111_return_constraint: 1.17562 w_k1_IBM_s11 - w_k1_IBM_s111
                                 + 0.99 b_k1_IBM_s111 - 1.01 s_k1_IBM_s111  = 0
 w_k1_IBM_s112_return_constraint: 1.5969 w_k1_IBM_s11 - w_k1_IBM_s112
                                 + 0.99 b_k1_IBM_s112 - 1.01 s_k1_IBM_s112  = 0
 w_k1_IBM_s113_return_constraint: 0.847992 w_k1_IBM_s11 - w_k1_IBM_s113
                                 + 0.99 b_k1_IBM_s113 - 1.01 s_k1_IBM_s113  = 0
 w_k1_IBM_s114_return_constraint: 0.657497 w_k1_IBM_s11 - w_k1_IBM_s114
                                 + 0.99 b_k1_IBM_s114 - 1.01 s_k1_IBM_s114  = 0
 w_k1_IBM_s115_return_constraint: 1.27154 w_k1_IBM_s11 - w_k1_IBM_s115
                                 + 0.99 b_k1_IBM_s115 - 1.01 s_k1_IBM_s115  = 0
 w_k1_IBM_s116_return_constraint: 1.0075 w_k1_IBM_s11 - w_k1_IBM_s116
                                 + 0.99 b_k1_IBM_s116 - 1.01 s_k1_IBM_s116  = 0
 w_k1_IBM_s117_return_constraint: 0.959764 w_k1_IBM_s11 - w_k1_IBM_s117
                                 + 0.99 b_k1_IBM_s117 - 1.01 s_k1_IBM_s117  = 0
 w_k1_INTC_s0_return_constraint: 1.0297 w_0_INTC - w_k1_INTC_s0
                                 + 0.99 b_k1_INTC_s0 - 1.01 s_k1_INTC_s0  = 0
 w_k1_INTC_s1_return_constraint: 0.969671 w_0_INTC - w_k1_INTC_s1
                                 + 0.99 b_k1_INTC_s1 - 1.01 s_k1_INTC_s1  = 0
 w_k1_INTC_s00_return_constraint: 1.15804 w_k1_INTC_s0 - w_k1_INTC_s00
                                 + 0.99 b_k1_INTC_s00 - 1.01 s_k1_INTC_s00  = 0
 w_k1_INTC_s01_return_constraint: 0.893913 w_k1_INTC_s0 - w_k1_INTC_s01
                                 + 0.99 b_k1_INTC_s01 - 1.01 s_k1_INTC_s01  = 0
 w_k1_INTC_s10_return_constraint: 0.961634 w_k1_INTC_s1 - w_k1_INTC_s10
                                 + 0.99 b_k1_INTC_s10 - 1.01 s_k1_INTC_s10  = 0
 w_k1_INTC_s11_return_constraint: 1.08159 w_k1_INTC_s1 - w_k1_INTC_s11
                                 + 0.99 b_k1_INTC_s11 - 1.01 s_k1_INTC_s11  = 0
 w_k1_INTC_s000_return_constraint: 1.05412 w_k1_INTC_s00 - w_k1_INTC_s000
                                 + 0.99 b_k1_INTC_s000 - 1.01 s_k1_INTC_s000
                                  = 0
 w_k1_INTC_s001_return_constraint: 1.15426 w_k1_INTC_s00 - w_k1_INTC_s001
                                 + 0.99 b_k1_INTC_s001 - 1.01 s_k1_INTC_s001
                                  = 0
 w_k1_INTC_s002_return_constraint: 0.911072 w_k1_INTC_s00 - w_k1_INTC_s002
                                 + 0.99 b_k1_INTC_s002 - 1.01 s_k1_INTC_s002
                                  = 0
 w_k1_INTC_s003_return_constraint: 1.11931 w_k1_INTC_s00 - w_k1_INTC_s003
                                 + 0.99 b_k1_INTC_s003 - 1.01 s_k1_INTC_s003
                                  = 0
 w_k1_INTC_s004_return_constraint: 0.892511 w_k1_INTC_s00 - w_k1_INTC_s004
                                 + 0.99 b_k1_INTC_s004 - 1.01 s_k1_INTC_s004
                                  = 0
 w_k1_INTC_s005_return_constraint: 1.18051 w_k1_INTC_s00 - w_k1_INTC_s005
                                 + 0.99 b_k1_INTC_s005 - 1.01 s_k1_INTC_s005
                                  = 0
 w_k1_INTC_s006_return_constraint: 1.21922 w_k1_INTC_s00 - w_k1_INTC_s006
                                 + 0.99 b_k1_INTC_s006 - 1.01 s_k1_INTC_s006
                                  = 0
 w_k1_INTC_s007_return_constraint: 1.42199 w_k1_INTC_s00 - w_k1_INTC_s007
                                 + 0.99 b_k1_INTC_s007 - 1.01 s_k1_INTC_s007
                                  = 0
 w_k1_INTC_s010_return_constraint: 0.99036 w_k1_INTC_s01 - w_k1_INTC_s010
                                 + 0.99 b_k1_INTC_s010 - 1.01 s_k1_INTC_s010
                                  = 0
 w_k1_INTC_s011_return_constraint: 0.819905 w_k1_INTC_s01 - w_k1_INTC_s011
                                 + 0.99 b_k1_INTC_s011 - 1.01 s_k1_INTC_s011
                                  = 0
 w_k1_INTC_s012_return_constraint: 1.37177 w_k1_INTC_s01 - w_k1_INTC_s012
                                 + 0.99 b_k1_INTC_s012 - 1.01 s_k1_INTC_s012
                                  = 0
 w_k1_INTC_s013_return_constraint: 1.08494 w_k1_INTC_s01 - w_k1_INTC_s013
                                 + 0.99 b_k1_INTC_s013 - 1.01 s_k1_INTC_s013
                                  = 0
 w_k1_INTC_s014_return_constraint: 1.25317 w_k1_INTC_s01 - w_k1_INTC_s014
                                 + 0.99 b_k1_INTC_s014 - 1.01 s_k1_INTC_s014
                                  = 0
 w_k1_INTC_s015_return_constraint: 1.46782 w_k1_INTC_s01 - w_k1_INTC_s015
                                 + 0.99 b_k1_INTC_s015 - 1.01 s_k1_INTC_s015
                                  = 0
 w_k1_INTC_s016_return_constraint: 1.03715 w_k1_INTC_s01 - w_k1_INTC_s016
                                 + 0.99 b_k1_INTC_s016 - 1.01 s_k1_INTC_s016
                                  = 0
 w_k1_INTC_s017_return_constraint: 0.931355 w_k1_INTC_s01 - w_k1_INTC_s017
                                 + 0.99 b_k1_INTC_s017 - 1.01 s_k1_INTC_s017
                                  = 0
 w_k1_INTC_s100_return_constraint: 1.21198 w_k1_INTC_s10 - w_k1_INTC_s100
                                 + 0.99 b_k1_INTC_s100 - 1.01 s_k1_INTC_s100
                                  = 0
 w_k1_INTC_s101_return_constraint: 1.60653 w_k1_INTC_s10 - w_k1_INTC_s101
                                 + 0.99 b_k1_INTC_s101 - 1.01 s_k1_INTC_s101
                                  = 0
 w_k1_INTC_s102_return_constraint: 0.720187 w_k1_INTC_s10 - w_k1_INTC_s102
                                 + 0.99 b_k1_INTC_s102 - 1.01 s_k1_INTC_s102
                                  = 0
 w_k1_INTC_s103_return_constraint: 1.09109 w_k1_INTC_s10 - w_k1_INTC_s103
                                 + 0.99 b_k1_INTC_s103 - 1.01 s_k1_INTC_s103
                                  = 0
 w_k1_INTC_s104_return_constraint: 1.02778 w_k1_INTC_s10 - w_k1_INTC_s104
                                 + 0.99 b_k1_INTC_s104 - 1.01 s_k1_INTC_s104
                                  = 0
 w_k1_INTC_s105_return_constraint: 1.43466 w_k1_INTC_s10 - w_k1_INTC_s105
                                 + 0.99 b_k1_INTC_s105 - 1.01 s_k1_INTC_s105
                                  = 0
 w_k1_INTC_s106_return_constraint: 0.843295 w_k1_INTC_s10 - w_k1_INTC_s106
                                 + 0.99 b_k1_INTC_s106 - 1.01 s_k1_INTC_s106
                                  = 0
 w_k1_INTC_s107_return_constraint: 0.766724 w_k1_INTC_s10 - w_k1_INTC_s107
                                 + 0.99 b_k1_INTC_s107 - 1.01 s_k1_INTC_s107
                                  = 0
 w_k1_INTC_s110_return_constraint: 1.04363 w_k1_INTC_s11 - w_k1_INTC_s110
                                 + 0.99 b_k1_INTC_s110 - 1.01 s_k1_INTC_s110
                                  = 0
 w_k1_INTC_s111_return_constraint: 1.05193 w_k1_INTC_s11 - w_k1_INTC_s111
                                 + 0.99 b_k1_INTC_s111 - 1.01 s_k1_INTC_s111
                                  = 0
 w_k1_INTC_s112_return_constraint: 0.840351 w_k1_INTC_s11 - w_k1_INTC_s112
                                 + 0.99 b_k1_INTC_s112 - 1.01 s_k1_INTC_s112
                                  = 0
 w_k1_INTC_s113_return_constraint: 0.787358 w_k1_INTC_s11 - w_k1_INTC_s113
                                 + 0.99 b_k1_INTC_s113 - 1.01 s_k1_INTC_s113
                                  = 0
 w_k1_INTC_s114_return_constraint: 0.833416 w_k1_INTC_s11 - w_k1_INTC_s114
                                 + 0.99 b_k1_INTC_s114 - 1.01 s_k1_INTC_s114
                                  = 0
 w_k1_INTC_s115_return_constraint: 1.14107 w_k1_INTC_s11 - w_k1_INTC_s115
                                 + 0.99 b_k1_INTC_s115 - 1.01 s_k1_INTC_s115
                                  = 0
 w_k1_INTC_s116_return_constraint: 1.01502 w_k1_INTC_s11 - w_k1_INTC_s116
                                 + 0.99 b_k1_INTC_s116 - 1.01 s_k1_INTC_s116
                                  = 0
 w_k1_INTC_s117_return_constraint: 0.830049 w_k1_INTC_s11 - w_k1_INTC_s117
                                 + 0.99 b_k1_INTC_s117 - 1.01 s_k1_INTC_s117
                                  = 0
 w_k1_KO_s0_return_constraint:   0.968406 w_0_KO - w_k1_KO_s0 + 0.99 b_k1_KO_s0
                                 - 1.01 s_k1_KO_s0  = 0
 w_k1_KO_s1_return_constraint:   1.09915 w_0_KO - w_k1_KO_s1 + 0.99 b_k1_KO_s1
                                 - 1.01 s_k1_KO_s1  = 0
 w_k1_KO_s00_return_constraint:  1.20439 w_k1_KO_s0 - w_k1_KO_s00
                                 + 0.99 b_k1_KO_s00 - 1.01 s_k1_KO_s00  = 0
 w_k1_KO_s01_return_constraint:  0.862696 w_k1_KO_s0 - w_k1_KO_s01
                                 + 0.99 b_k1_KO_s01 - 1.01 s_k1_KO_s01  = 0
 w_k1_KO_s10_return_constraint:  0.854094 w_k1_KO_s1 - w_k1_KO_s10
                                 + 0.99 b_k1_KO_s10 - 1.01 s_k1_KO_s10  = 0
 w_k1_KO_s11_return_constraint:  1.15268 w_k1_KO_s1 - w_k1_KO_s11
                                 + 0.99 b_k1_KO_s11 - 1.01 s_k1_KO_s11  = 0
 w_k1_KO_s000_return_constraint: 1.0817 w_k1_KO_s00 - w_k1_KO_s000
                                 + 0.99 b_k1_KO_s000 - 1.01 s_k1_KO_s000  = 0
 w_k1_KO_s001_return_constraint: 1.02435 w_k1_KO_s00 - w_k1_KO_s001
                                 + 0.99 b_k1_KO_s001 - 1.01 s_k1_KO_s001  = 0
 w_k1_KO_s002_return_constraint: 0.917472 w_k1_KO_s00 - w_k1_KO_s002
                                 + 0.99 b_k1_KO_s002 - 1.01 s_k1_KO_s002  = 0
 w_k1_KO_s003_return_constraint: 0.82004 w_k1_KO_s00 - w_k1_KO_s003
                                 + 0.99 b_k1_KO_s003 - 1.01 s_k1_KO_s003  = 0
 w_k1_KO_s004_return_constraint: 0.691464 w_k1_KO_s00 - w_k1_KO_s004
                                 + 0.99 b_k1_KO_s004 - 1.01 s_k1_KO_s004  = 0
 w_k1_KO_s005_return_constraint: 1.07995 w_k1_KO_s00 - w_k1_KO_s005
                                 + 0.99 b_k1_KO_s005 - 1.01 s_k1_KO_s005  = 0
 w_k1_KO_s006_return_constraint: 1.39212 w_k1_KO_s00 - w_k1_KO_s006
                                 + 0.99 b_k1_KO_s006 - 1.01 s_k1_KO_s006  = 0
 w_k1_KO_s007_return_constraint: 1.5911 w_k1_KO_s00 - w_k1_KO_s007
                                 + 0.99 b_k1_KO_s007 - 1.01 s_k1_KO_s007  = 0
 w_k1_KO_s010_return_constraint: 0.902108 w_k1_KO_s01 - w_k1_KO_s010
                                 + 0.99 b_k1_KO_s010 - 1.01 s_k1_KO_s010  = 0
 w_k1_KO_s011_return_constraint: 0.768574 w_k1_KO_s01 - w_k1_KO_s011
                                 + 0.99 b_k1_KO_s011 - 1.01 s_k1_KO_s011  = 0
 w_k1_KO_s012_return_constraint: 1.22277 w_k1_KO_s01 - w_k1_KO_s012
                                 + 0.99 b_k1_KO_s012 - 1.01 s_k1_KO_s012  = 0
 w_k1_KO_s013_return_constraint: 1.08757 w_k1_KO_s01 - w_k1_KO_s013
                                 + 0.99 b_k1_KO_s013 - 1.01 s_k1_KO_s013  = 0
 w_k1_KO_s014_return_constraint: 1.5022 w_k1_KO_s01 - w_k1_KO_s014
                                 + 0.99 b_k1_KO_s014 - 1.01 s_k1_KO_s014  = 0
 w_k1_KO_s015_return_constraint: 1.84011 w_k1_KO_s01 - w_k1_KO_s015
                                 + 0.99 b_k1_KO_s015 - 1.01 s_k1_KO_s015  = 0
 w_k1_KO_s016_return_constraint: 1.20674 w_k1_KO_s01 - w_k1_KO_s016
                                 + 0.99 b_k1_KO_s016 - 1.01 s_k1_KO_s016  = 0
 w_k1_KO_s017_return_constraint: 1.22135 w_k1_KO_s01 - w_k1_KO_s017
                                 + 0.99 b_k1_KO_s017 - 1.01 s_k1_KO_s017  = 0
 w_k1_KO_s100_return_constraint: 1.31338 w_k1_KO_s10 - w_k1_KO_s100
                                 + 0.99 b_k1_KO_s100 - 1.01 s_k1_KO_s100  = 0
 w_k1_KO_s101_return_constraint: 1.67361 w_k1_KO_s10 - w_k1_KO_s101
                                 + 0.99 b_k1_KO_s101 - 1.01 s_k1_KO_s101  = 0
 w_k1_KO_s102_return_constraint: 0.587303 w_k1_KO_s10 - w_k1_KO_s102
                                 + 0.99 b_k1_KO_s102 - 1.01 s_k1_KO_s102  = 0
 w_k1_KO_s103_return_constraint: 0.93284 w_k1_KO_s10 - w_k1_KO_s103
                                 + 0.99 b_k1_KO_s103 - 1.01 s_k1_KO_s103  = 0
 w_k1_KO_s104_return_constraint: 1.03921 w_k1_KO_s10 - w_k1_KO_s104
                                 + 0.99 b_k1_KO_s104 - 1.01 s_k1_KO_s104  = 0
 w_k1_KO_s105_return_constraint: 1.3466 w_k1_KO_s10 - w_k1_KO_s105
                                 + 0.99 b_k1_KO_s105 - 1.01 s_k1_KO_s105  = 0
 w_k1_KO_s106_return_constraint: 0.700433 w_k1_KO_s10 - w_k1_KO_s106
                                 + 0.99 b_k1_KO_s106 - 1.01 s_k1_KO_s106  = 0
 w_k1_KO_s107_return_constraint: 0.797432 w_k1_KO_s10 - w_k1_KO_s107
                                 + 0.99 b_k1_KO_s107 - 1.01 s_k1_KO_s107  = 0
 w_k1_KO_s110_return_constraint: 0.950421 w_k1_KO_s11 - w_k1_KO_s110
                                 + 0.99 b_k1_KO_s110 - 1.01 s_k1_KO_s110  = 0
 w_k1_KO_s111_return_constraint: 0.99663 w_k1_KO_s11 - w_k1_KO_s111
                                 + 0.99 b_k1_KO_s111 - 1.01 s_k1_KO_s111  = 0
 w_k1_KO_s112_return_constraint: 1.18279 w_k1_KO_s11 - w_k1_KO_s112
                                 + 0.99 b_k1_KO_s112 - 1.01 s_k1_KO_s112  = 0
 w_k1_KO_s113_return_constraint: 0.689949 w_k1_KO_s11 - w_k1_KO_s113
                                 + 0.99 b_k1_KO_s113 - 1.01 s_k1_KO_s113  = 0
 w_k1_KO_s114_return_constraint: 0.599222 w_k1_KO_s11 - w_k1_KO_s114
                                 + 0.99 b_k1_KO_s114 - 1.01 s_k1_KO_s114  = 0
 w_k1_KO_s115_return_constraint: 1.45819 w_k1_KO_s11 - w_k1_KO_s115
                                 + 0.99 b_k1_KO_s115 - 1.01 s_k1_KO_s115  = 0
 w_k1_KO_s116_return_constraint: 1.18043 w_k1_KO_s11 - w_k1_KO_s116
                                 + 0.99 b_k1_KO_s116 - 1.01 s_k1_KO_s116  = 0
 w_k1_KO_s117_return_constraint: 0.809301 w_k1_KO_s11 - w_k1_KO_s117
                                 + 0.99 b_k1_KO_s117 - 1.01 s_k1_KO_s117  = 0
 w_k1_MAT_s0_return_constraint:  0.895595 w_0_MAT - w_k1_MAT_s0
                                 + 0.99 b_k1_MAT_s0 - 1.01 s_k1_MAT_s0  = 0
 w_k1_MAT_s1_return_constraint:  1.15582 w_0_MAT - w_k1_MAT_s1
                                 + 0.99 b_k1_MAT_s1 - 1.01 s_k1_MAT_s1  = 0
 w_k1_MAT_s00_return_constraint: 1.12959 w_k1_MAT_s0 - w_k1_MAT_s00
                                 + 0.99 b_k1_MAT_s00 - 1.01 s_k1_MAT_s00  = 0
 w_k1_MAT_s01_return_constraint: 0.931451 w_k1_MAT_s0 - w_k1_MAT_s01
                                 + 0.99 b_k1_MAT_s01 - 1.01 s_k1_MAT_s01  = 0
 w_k1_MAT_s10_return_constraint: 0.891137 w_k1_MAT_s1 - w_k1_MAT_s10
                                 + 0.99 b_k1_MAT_s10 - 1.01 s_k1_MAT_s10  = 0
 w_k1_MAT_s11_return_constraint: 1.40316 w_k1_MAT_s1 - w_k1_MAT_s11
                                 + 0.99 b_k1_MAT_s11 - 1.01 s_k1_MAT_s11  = 0
 w_k1_MAT_s000_return_constraint: 1.28006 w_k1_MAT_s00 - w_k1_MAT_s000
                                 + 0.99 b_k1_MAT_s000 - 1.01 s_k1_MAT_s000  = 0
 w_k1_MAT_s001_return_constraint: 1.00981 w_k1_MAT_s00 - w_k1_MAT_s001
                                 + 0.99 b_k1_MAT_s001 - 1.01 s_k1_MAT_s001  = 0
 w_k1_MAT_s002_return_constraint: 0.888062 w_k1_MAT_s00 - w_k1_MAT_s002
                                 + 0.99 b_k1_MAT_s002 - 1.01 s_k1_MAT_s002  = 0
 w_k1_MAT_s003_return_constraint: 0.620227 w_k1_MAT_s00 - w_k1_MAT_s003
                                 + 0.99 b_k1_MAT_s003 - 1.01 s_k1_MAT_s003  = 0
 w_k1_MAT_s004_return_constraint: 0.602539 w_k1_MAT_s00 - w_k1_MAT_s004
                                 + 0.99 b_k1_MAT_s004 - 1.01 s_k1_MAT_s004  = 0
 w_k1_MAT_s005_return_constraint: 0.937504 w_k1_MAT_s00 - w_k1_MAT_s005
                                 + 0.99 b_k1_MAT_s005 - 1.01 s_k1_MAT_s005  = 0
 w_k1_MAT_s006_return_constraint: 1.38101 w_k1_MAT_s00 - w_k1_MAT_s006
                                 + 0.99 b_k1_MAT_s006 - 1.01 s_k1_MAT_s006  = 0
 w_k1_MAT_s007_return_constraint: 1.55794 w_k1_MAT_s00 - w_k1_MAT_s007
                                 + 0.99 b_k1_MAT_s007 - 1.01 s_k1_MAT_s007  = 0
 w_k1_MAT_s010_return_constraint: 0.715957 w_k1_MAT_s01 - w_k1_MAT_s010
                                 + 0.99 b_k1_MAT_s010 - 1.01 s_k1_MAT_s010  = 0
 w_k1_MAT_s011_return_constraint: 0.708221 w_k1_MAT_s01 - w_k1_MAT_s011
                                 + 0.99 b_k1_MAT_s011 - 1.01 s_k1_MAT_s011  = 0
 w_k1_MAT_s012_return_constraint: 1.57936 w_k1_MAT_s01 - w_k1_MAT_s012
                                 + 0.99 b_k1_MAT_s012 - 1.01 s_k1_MAT_s012  = 0
 w_k1_MAT_s013_return_constraint: 0.905879 w_k1_MAT_s01 - w_k1_MAT_s013
                                 + 0.99 b_k1_MAT_s013 - 1.01 s_k1_MAT_s013  = 0
 w_k1_MAT_s014_return_constraint: 1.74242 w_k1_MAT_s01 - w_k1_MAT_s014
                                 + 0.99 b_k1_MAT_s014 - 1.01 s_k1_MAT_s014  = 0
 w_k1_MAT_s015_return_constraint: 2.27304 w_k1_MAT_s01 - w_k1_MAT_s015
                                 + 0.99 b_k1_MAT_s015 - 1.01 s_k1_MAT_s015  = 0
 w_k1_MAT_s016_return_constraint: 1.08323 w_k1_MAT_s01 - w_k1_MAT_s016
                                 + 0.99 b_k1_MAT_s016 - 1.01 s_k1_MAT_s016  = 0
 w_k1_MAT_s017_return_constraint: 1.48581 w_k1_MAT_s01 - w_k1_MAT_s017
                                 + 0.99 b_k1_MAT_s017 - 1.01 s_k1_MAT_s017  = 0
 w_k1_MAT_s100_return_constraint: 1.22564 w_k1_MAT_s10 - w_k1_MAT_s100
                                 + 0.99 b_k1_MAT_s100 - 1.01 s_k1_MAT_s100  = 0
 w_k1_MAT_s101_return_constraint: 1.81169 w_k1_MAT_s10 - w_k1_MAT_s101
                                 + 0.99 b_k1_MAT_s101 - 1.01 s_k1_MAT_s101  = 0
 w_k1_MAT_s102_return_constraint: 0.390592 w_k1_MAT_s10 - w_k1_MAT_s102
                                 + 0.99 b_k1_MAT_s102 - 1.01 s_k1_MAT_s102  = 0
 w_k1_MAT_s103_return_constraint: 0.683733 w_k1_MAT_s10 - w_k1_MAT_s103
                                 + 0.99 b_k1_MAT_s103 - 1.01 s_k1_MAT_s103  = 0
 w_k1_MAT_s104_return_constraint: 1.02398 w_k1_MAT_s10 - w_k1_MAT_s104
                                 + 0.99 b_k1_MAT_s104 - 1.01 s_k1_MAT_s104  = 0
 w_k1_MAT_s105_return_constraint: 1.62968 w_k1_MAT_s10 - w_k1_MAT_s105
                                 + 0.99 b_k1_MAT_s105 - 1.01 s_k1_MAT_s105  = 0
 w_k1_MAT_s106_return_constraint: 0.68496 w_k1_MAT_s10 - w_k1_MAT_s106
                                 + 0.99 b_k1_MAT_s106 - 1.01 s_k1_MAT_s106  = 0
 w_k1_MAT_s107_return_constraint: 0.701278 w_k1_MAT_s10 - w_k1_MAT_s107
                                 + 0.99 b_k1_MAT_s107 - 1.01 s_k1_MAT_s107  = 0
 w_k1_MAT_s110_return_constraint: 0.877192 w_k1_MAT_s11 - w_k1_MAT_s110
                                 + 0.99 b_k1_MAT_s110 - 1.01 s_k1_MAT_s110  = 0
 w_k1_MAT_s111_return_constraint: 1.24166 w_k1_MAT_s11 - w_k1_MAT_s111
                                 + 0.99 b_k1_MAT_s111 - 1.01 s_k1_MAT_s111  = 0
 w_k1_MAT_s112_return_constraint: 1.45146 w_k1_MAT_s11 - w_k1_MAT_s112
                                 + 0.99 b_k1_MAT_s112 - 1.01 s_k1_MAT_s112  = 0
 w_k1_MAT_s113_return_constraint: 0.730861 w_k1_MAT_s11 - w_k1_MAT_s113
                                 + 0.99 b_k1_MAT_s113 - 1.01 s_k1_MAT_s113  = 0
 w_k1_MAT_s114_return_constraint: 0.5443 w_k1_MAT_s11 - w_k1_MAT_s114
                                 + 0.99 b_k1_MAT_s114 - 1.01 s_k1_MAT_s114  = 0
 w_k1_MAT_s115_return_constraint: 1.53281 w_k1_MAT_s11 - w_k1_MAT_s115
                                 + 0.99 b_k1_MAT_s115 - 1.01 s_k1_MAT_s115  = 0
 w_k1_MAT_s116_return_constraint: 1.00536 w_k1_MAT_s11 - w_k1_MAT_s116
                                 + 0.99 b_k1_MAT_s116 - 1.01 s_k1_MAT_s116  = 0
 w_k1_MAT_s117_return_constraint: 0.7014 w_k1_MAT_s11 - w_k1_MAT_s117
                                 + 0.99 b_k1_MAT_s117 - 1.01 s_k1_MAT_s117  = 0
 w_k1_MSFT_s0_return_constraint: 1.03446 w_0_MSFT - w_k1_MSFT_s0
                                 + 0.99 b_k1_MSFT_s0 - 1.01 s_k1_MSFT_s0  = 0
 w_k1_MSFT_s1_return_constraint: 1.00281 w_0_MSFT - w_k1_MSFT_s1
                                 + 0.99 b_k1_MSFT_s1 - 1.01 s_k1_MSFT_s1  = 0
 w_k1_MSFT_s00_return_constraint: 1.19297 w_k1_MSFT_s0 - w_k1_MSFT_s00
                                 + 0.99 b_k1_MSFT_s00 - 1.01 s_k1_MSFT_s00  = 0
 w_k1_MSFT_s01_return_constraint: 0.84655 w_k1_MSFT_s0 - w_k1_MSFT_s01
                                 + 0.99 b_k1_MSFT_s01 - 1.01 s_k1_MSFT_s01  = 0
 w_k1_MSFT_s10_return_constraint: 0.963747 w_k1_MSFT_s1 - w_k1_MSFT_s10
                                 + 0.99 b_k1_MSFT_s10 - 1.01 s_k1_MSFT_s10  = 0
 w_k1_MSFT_s11_return_constraint: 1.06343 w_k1_MSFT_s1 - w_k1_MSFT_s11
                                 + 0.99 b_k1_MSFT_s11 - 1.01 s_k1_MSFT_s11  = 0
 w_k1_MSFT_s000_return_constraint: 0.951675 w_k1_MSFT_s00 - w_k1_MSFT_s000
                                 + 0.99 b_k1_MSFT_s000 - 1.01 s_k1_MSFT_s000
                                  = 0
 w_k1_MSFT_s001_return_constraint: 1.2703 w_k1_MSFT_s00 - w_k1_MSFT_s001
                                 + 0.99 b_k1_MSFT_s001 - 1.01 s_k1_MSFT_s001
                                  = 0
 w_k1_MSFT_s002_return_constraint: 0.884051 w_k1_MSFT_s00 - w_k1_MSFT_s002
                                 + 0.99 b_k1_MSFT_s002 - 1.01 s_k1_MSFT_s002
                                  = 0
 w_k1_MSFT_s003_return_constraint: 1.08386 w_k1_MSFT_s00 - w_k1_MSFT_s003
                                 + 0.99 b_k1_MSFT_s003 - 1.01 s_k1_MSFT_s003
                                  = 0
 w_k1_MSFT_s004_return_constraint: 0.809029 w_k1_MSFT_s00 - w_k1_MSFT_s004
                                 + 0.99 b_k1_MSFT_s004 - 1.01 s_k1_MSFT_s004
                                  = 0
 w_k1_MSFT_s005_return_constraint: 1.12355 w_k1_MSFT_s00 - w_k1_MSFT_s005
                                 + 0.99 b_k1_MSFT_s005 - 1.01 s_k1_MSFT_s005
                                  = 0
 w_k1_MSFT_s006_return_constraint: 1.19814 w_k1_MSFT_s00 - w_k1_MSFT_s006
                                 + 0.99 b_k1_MSFT_s006 - 1.01 s_k1_MSFT_s006
                                  = 0
 w_k1_MSFT_s007_return_constraint: 1.33697 w_k1_MSFT_s00 - w_k1_MSFT_s007
                                 + 0.99 b_k1_MSFT_s007 - 1.01 s_k1_MSFT_s007
                                  = 0
 w_k1_MSFT_s010_return_constraint: 0.942432 w_k1_MSFT_s01 - w_k1_MSFT_s010
                                 + 0.99 b_k1_MSFT_s010 - 1.01 s_k1_MSFT_s010
                                  = 0
 w_k1_MSFT_s011_return_constraint: 0.786128 w_k1_MSFT_s01 - w_k1_MSFT_s011
                                 + 0.99 b_k1_MSFT_s011 - 1.01 s_k1_MSFT_s011
                                  = 0
 w_k1_MSFT_s012_return_constraint: 1.29863 w_k1_MSFT_s01 - w_k1_MSFT_s012
                                 + 0.99 b_k1_MSFT_s012 - 1.01 s_k1_MSFT_s012
                                  = 0
 w_k1_MSFT_s013_return_constraint: 1.28767 w_k1_MSFT_s01 - w_k1_MSFT_s013
                                 + 0.99 b_k1_MSFT_s013 - 1.01 s_k1_MSFT_s013
                                  = 0
 w_k1_MSFT_s014_return_constraint: 1.27053 w_k1_MSFT_s01 - w_k1_MSFT_s014
                                 + 0.99 b_k1_MSFT_s014 - 1.01 s_k1_MSFT_s014
                                  = 0
 w_k1_MSFT_s015_return_constraint: 1.61618 w_k1_MSFT_s01 - w_k1_MSFT_s015
                                 + 0.99 b_k1_MSFT_s015 - 1.01 s_k1_MSFT_s015
                                  = 0
 w_k1_MSFT_s016_return_constraint: 1.07099 w_k1_MSFT_s01 - w_k1_MSFT_s016
                                 + 0.99 b_k1_MSFT_s016 - 1.01 s_k1_MSFT_s016
                                  = 0
 w_k1_MSFT_s017_return_constraint: 1.03446 w_k1_MSFT_s01 - w_k1_MSFT_s017
                                 + 0.99 b_k1_MSFT_s017 - 1.01 s_k1_MSFT_s017
                                  = 0
 w_k1_MSFT_s100_return_constraint: 1.10378 w_k1_MSFT_s10 - w_k1_MSFT_s100
                                 + 0.99 b_k1_MSFT_s100 - 1.01 s_k1_MSFT_s100
                                  = 0
 w_k1_MSFT_s101_return_constraint: 1.45904 w_k1_MSFT_s10 - w_k1_MSFT_s101
                                 + 0.99 b_k1_MSFT_s101 - 1.01 s_k1_MSFT_s101
                                  = 0
 w_k1_MSFT_s102_return_constraint: 0.680867 w_k1_MSFT_s10 - w_k1_MSFT_s102
                                 + 0.99 b_k1_MSFT_s102 - 1.01 s_k1_MSFT_s102
                                  = 0
 w_k1_MSFT_s103_return_constraint: 1.01797 w_k1_MSFT_s10 - w_k1_MSFT_s103
                                 + 0.99 b_k1_MSFT_s103 - 1.01 s_k1_MSFT_s103
                                  = 0
 w_k1_MSFT_s104_return_constraint: 1.04746 w_k1_MSFT_s10 - w_k1_MSFT_s104
                                 + 0.99 b_k1_MSFT_s104 - 1.01 s_k1_MSFT_s104
                                  = 0
 w_k1_MSFT_s105_return_constraint: 1.45007 w_k1_MSFT_s10 - w_k1_MSFT_s105
                                 + 0.99 b_k1_MSFT_s105 - 1.01 s_k1_MSFT_s105
                                  = 0
 w_k1_MSFT_s106_return_constraint: 0.713003 w_k1_MSFT_s10 - w_k1_MSFT_s106
                                 + 0.99 b_k1_MSFT_s106 - 1.01 s_k1_MSFT_s106
                                  = 0
 w_k1_MSFT_s107_return_constraint: 0.806738 w_k1_MSFT_s10 - w_k1_MSFT_s107
                                 + 0.99 b_k1_MSFT_s107 - 1.01 s_k1_MSFT_s107
                                  = 0
 w_k1_MSFT_s110_return_constraint: 1.05818 w_k1_MSFT_s11 - w_k1_MSFT_s110
                                 + 0.99 b_k1_MSFT_s110 - 1.01 s_k1_MSFT_s110
                                  = 0
 w_k1_MSFT_s111_return_constraint: 0.863281 w_k1_MSFT_s11 - w_k1_MSFT_s111
                                 + 0.99 b_k1_MSFT_s111 - 1.01 s_k1_MSFT_s111
                                  = 0
 w_k1_MSFT_s112_return_constraint: 0.865962 w_k1_MSFT_s11 - w_k1_MSFT_s112
                                 + 0.99 b_k1_MSFT_s112 - 1.01 s_k1_MSFT_s112
                                  = 0
 w_k1_MSFT_s113_return_constraint: 0.769832 w_k1_MSFT_s11 - w_k1_MSFT_s113
                                 + 0.99 b_k1_MSFT_s113 - 1.01 s_k1_MSFT_s113
                                  = 0
 w_k1_MSFT_s114_return_constraint: 0.693061 w_k1_MSFT_s11 - w_k1_MSFT_s114
                                 + 0.99 b_k1_MSFT_s114 - 1.01 s_k1_MSFT_s114
                                  = 0
 w_k1_MSFT_s115_return_constraint: 1.36277 w_k1_MSFT_s11 - w_k1_MSFT_s115
                                 + 0.99 b_k1_MSFT_s115 - 1.01 s_k1_MSFT_s115
                                  = 0
 w_k1_MSFT_s116_return_constraint: 1.02413 w_k1_MSFT_s11 - w_k1_MSFT_s116
                                 + 0.99 b_k1_MSFT_s116 - 1.01 s_k1_MSFT_s116
                                  = 0
 w_k1_MSFT_s117_return_constraint: 0.694579 w_k1_MSFT_s11 - w_k1_MSFT_s117
                                 + 0.99 b_k1_MSFT_s117 - 1.01 s_k1_MSFT_s117
                                  = 0
 w_k1_PG_s0_return_constraint:   0.991305 w_0_PG - w_k1_PG_s0 + 0.99 b_k1_PG_s0
                                 - 1.01 s_k1_PG_s0  = 0
 w_k1_PG_s1_return_constraint:   1.00457 w_0_PG - w_k1_PG_s1 + 0.99 b_k1_PG_s1
                                 - 1.01 s_k1_PG_s1  = 0
 w_k1_PG_s00_return_constraint:  1.18303 w_k1_PG_s0 - w_k1_PG_s00
                                 + 0.99 b_k1_PG_s00 - 1.01 s_k1_PG_s00  = 0
 w_k1_PG_s01_return_constraint:  0.911259 w_k1_PG_s0 - w_k1_PG_s01
                                 + 0.99 b_k1_PG_s01 - 1.01 s_k1_PG_s01  = 0
 w_k1_PG_s10_return_constraint:  0.979127 w_k1_PG_s1 - w_k1_PG_s10
                                 + 0.99 b_k1_PG_s10 - 1.01 s_k1_PG_s10  = 0
 w_k1_PG_s11_return_constraint:  1.09487 w_k1_PG_s1 - w_k1_PG_s11
                                 + 0.99 b_k1_PG_s11 - 1.01 s_k1_PG_s11  = 0
 w_k1_PG_s000_return_constraint: 0.966357 w_k1_PG_s00 - w_k1_PG_s000
                                 + 0.99 b_k1_PG_s000 - 1.01 s_k1_PG_s000  = 0
 w_k1_PG_s001_return_constraint: 1.04815 w_k1_PG_s00 - w_k1_PG_s001
                                 + 0.99 b_k1_PG_s001 - 1.01 s_k1_PG_s001  = 0
 w_k1_PG_s002_return_constraint: 0.908326 w_k1_PG_s00 - w_k1_PG_s002
                                 + 0.99 b_k1_PG_s002 - 1.01 s_k1_PG_s002  = 0
 w_k1_PG_s003_return_constraint: 1.0381 w_k1_PG_s00 - w_k1_PG_s003
                                 + 0.99 b_k1_PG_s003 - 1.01 s_k1_PG_s003  = 0
 w_k1_PG_s004_return_constraint: 0.763099 w_k1_PG_s00 - w_k1_PG_s004
                                 + 0.99 b_k1_PG_s004 - 1.01 s_k1_PG_s004  = 0
 w_k1_PG_s005_return_constraint: 1.05702 w_k1_PG_s00 - w_k1_PG_s005
                                 + 0.99 b_k1_PG_s005 - 1.01 s_k1_PG_s005  = 0
 w_k1_PG_s006_return_constraint: 1.11544 w_k1_PG_s00 - w_k1_PG_s006
                                 + 0.99 b_k1_PG_s006 - 1.01 s_k1_PG_s006  = 0
 w_k1_PG_s007_return_constraint: 1.28479 w_k1_PG_s00 - w_k1_PG_s007
                                 + 0.99 b_k1_PG_s007 - 1.01 s_k1_PG_s007  = 0
 w_k1_PG_s010_return_constraint: 0.992513 w_k1_PG_s01 - w_k1_PG_s010
                                 + 0.99 b_k1_PG_s010 - 1.01 s_k1_PG_s010  = 0
 w_k1_PG_s011_return_constraint: 0.823559 w_k1_PG_s01 - w_k1_PG_s011
                                 + 0.99 b_k1_PG_s011 - 1.01 s_k1_PG_s011  = 0
 w_k1_PG_s012_return_constraint: 1.28439 w_k1_PG_s01 - w_k1_PG_s012
                                 + 0.99 b_k1_PG_s012 - 1.01 s_k1_PG_s012  = 0
 w_k1_PG_s013_return_constraint: 1.15659 w_k1_PG_s01 - w_k1_PG_s013
                                 + 0.99 b_k1_PG_s013 - 1.01 s_k1_PG_s013  = 0
 w_k1_PG_s014_return_constraint: 1.23436 w_k1_PG_s01 - w_k1_PG_s014
                                 + 0.99 b_k1_PG_s014 - 1.01 s_k1_PG_s014  = 0
 w_k1_PG_s015_return_constraint: 1.43675 w_k1_PG_s01 - w_k1_PG_s015
                                 + 0.99 b_k1_PG_s015 - 1.01 s_k1_PG_s015  = 0
 w_k1_PG_s016_return_constraint: 1.11876 w_k1_PG_s01 - w_k1_PG_s016
                                 + 0.99 b_k1_PG_s016 - 1.01 s_k1_PG_s016  = 0
 w_k1_PG_s017_return_constraint: 1.04629 w_k1_PG_s01 - w_k1_PG_s017
                                 + 0.99 b_k1_PG_s017 - 1.01 s_k1_PG_s017  = 0
 w_k1_PG_s100_return_constraint: 1.0569 w_k1_PG_s10 - w_k1_PG_s100
                                 + 0.99 b_k1_PG_s100 - 1.01 s_k1_PG_s100  = 0
 w_k1_PG_s101_return_constraint: 1.34323 w_k1_PG_s10 - w_k1_PG_s101
                                 + 0.99 b_k1_PG_s101 - 1.01 s_k1_PG_s101  = 0
 w_k1_PG_s102_return_constraint: 0.707233 w_k1_PG_s10 - w_k1_PG_s102
                                 + 0.99 b_k1_PG_s102 - 1.01 s_k1_PG_s102  = 0
 w_k1_PG_s103_return_constraint: 0.937701 w_k1_PG_s10 - w_k1_PG_s103
                                 + 0.99 b_k1_PG_s103 - 1.01 s_k1_PG_s103  = 0
 w_k1_PG_s104_return_constraint: 0.949075 w_k1_PG_s10 - w_k1_PG_s104
                                 + 0.99 b_k1_PG_s104 - 1.01 s_k1_PG_s104  = 0
 w_k1_PG_s105_return_constraint: 1.29142 w_k1_PG_s10 - w_k1_PG_s105
                                 + 0.99 b_k1_PG_s105 - 1.01 s_k1_PG_s105  = 0
 w_k1_PG_s106_return_constraint: 0.837296 w_k1_PG_s10 - w_k1_PG_s106
                                 + 0.99 b_k1_PG_s106 - 1.01 s_k1_PG_s106  = 0
 w_k1_PG_s107_return_constraint: 0.820299 w_k1_PG_s10 - w_k1_PG_s107
                                 + 0.99 b_k1_PG_s107 - 1.01 s_k1_PG_s107  = 0
 w_k1_PG_s110_return_constraint: 1.05988 w_k1_PG_s11 - w_k1_PG_s110
                                 + 0.99 b_k1_PG_s110 - 1.01 s_k1_PG_s110  = 0
 w_k1_PG_s111_return_constraint: 1.05356 w_k1_PG_s11 - w_k1_PG_s111
                                 + 0.99 b_k1_PG_s111 - 1.01 s_k1_PG_s111  = 0
 w_k1_PG_s112_return_constraint: 1.0222 w_k1_PG_s11 - w_k1_PG_s112
                                 + 0.99 b_k1_PG_s112 - 1.01 s_k1_PG_s112  = 0
 w_k1_PG_s113_return_constraint: 0.883058 w_k1_PG_s11 - w_k1_PG_s113
                                 + 0.99 b_k1_PG_s113 - 1.01 s_k1_PG_s113  = 0
 w_k1_PG_s114_return_constraint: 0.698696 w_k1_PG_s11 - w_k1_PG_s114
                                 + 0.99 b_k1_PG_s114 - 1.01 s_k1_PG_s114  = 0
 w_k1_PG_s115_return_constraint: 1.27636 w_k1_PG_s11 - w_k1_PG_s115
                                 + 0.99 b_k1_PG_s115 - 1.01 s_k1_PG_s115  = 0
 w_k1_PG_s116_return_constraint: 1.03283 w_k1_PG_s11 - w_k1_PG_s116
                                 + 0.99 b_k1_PG_s116 - 1.01 s_k1_PG_s116  = 0
 w_k1_PG_s117_return_constraint: 0.796385 w_k1_PG_s11 - w_k1_PG_s117
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
 k1_wcvar_constraint:            wcvar - var_1 - 0.165999962108516 z_k1_s0000
                                 - 0.118000162307461 z_k1_s0001
                                 - 0.168999638326998 z_k1_s0002
                                 - 0.23899974543313 z_k1_s0003
                                 - 0.147000735723427 z_k1_s0004
                                 - 0.149000519869081 z_k1_s0005
                                 - 0.0570001054578705 z_k1_s0006
                                 - 0.232000500923339 z_k1_s0007
                                 - 0.617998961300426 z_k1_s0010
                                 - 0.200999717603447 z_k1_s0011
                                 - 0.222999722201776 z_k1_s0012
                                 - 0.240000266435969 z_k1_s0013
                                 - 0.321999504994645 z_k1_s0014
                                 - 0.199999587631006 z_k1_s0015
                                 - 0.400998829435227 z_k1_s0016
                                 - 0.173999776841755 z_k1_s0017
                                 - 0.534999958630078 z_k1_s0020
                                 - 0.437999270148562 z_k1_s0021
                                 - 0.748001365759452 z_k1_s0022
                                 - 0.256000442083415 z_k1_s0023
                                 - 0.259000233056765 z_k1_s0024
                                 - 0.351999846238687 z_k1_s0025
                                 - 0.220000242399628 z_k1_s0026
                                 - 0.577001094262359 z_k1_s0027
                                 - 0.189000128154917 z_k1_s0030
                                 - 0.233000336952956 z_k1_s0031
                                 - 0.162000109847072 z_k1_s0032
                                 - 0.0709999855057916 z_k1_s0033
                                 - 0.132999642775512 z_k1_s0034
                                 - 0.130000177630401 z_k1_s0035
                                 - 0.179000300337265 z_k1_s0036
                                 - 0.111000021576677 z_k1_s0037
                                 - 0.864000902599967 z_k1_s0040
                                 - 0.582999567611157 z_k1_s0041
                                 - 0.865998363806299 z_k1_s0042
                                 - 0.379999563828929 z_k1_s0043
                                 - 0.86299746100339 z_k1_s0044
                                 - 0.204999820485265 z_k1_s0045
                                 - 0.468999180253203 z_k1_s0046
                                 - 0.480997138272743 z_k1_s0047
                                 - 0.282000715108259 z_k1_s0050
                                 - 0.160000649063188 z_k1_s0051
                                 - 0.0960004746782825 z_k1_s0052
                                 - 0.43700180338709 z_k1_s0053
                                 - 0.331001141635473 z_k1_s0054
                                 - 0.279000254094502 z_k1_s0055
                                 - 0.0980004268525797 z_k1_s0056
                                 - 0.448002073098035 z_k1_s0057
                                 - 0.202999968637967 z_k1_s0060
                                 - 0.236999665109294 z_k1_s0061
                                 - 0.194000506925056 z_k1_s0062
                                 - 0.242999306251235 z_k1_s0063
                                 - 0.138000050266218 z_k1_s0064
                                 - 0.226000323015736 z_k1_s0065
                                 - 0.133999943504858 z_k1_s0066
                                 - 0.354999527568779 z_k1_s0067
                                 - 0.128000435962381 z_k1_s0070
                                 - 0.353000381675221 z_k1_s0071
                                 - 0.373002237299314 z_k1_s0072
                                 - 0.322000495466192 z_k1_s0073
                                 - 0.574000996266134 z_k1_s0074
                                 - 0.325000903810167 z_k1_s0075
                                 - 0.38200086232401 z_k1_s0076
                                 - 0.143000397676474 z_k1_s0077
                                 - 0.830998978713943 z_k1_s0100
                                 - 1.24999844865998 z_k1_s0101
                                 - 0.932997516167881 z_k1_s0102
                                 - 0.497999211184752 z_k1_s0103
                                 - 1.03999961331597 z_k1_s0104
                                 - 0.569998991245331 z_k1_s0105
                                 - 0.893000428575154 z_k1_s0106
                                 - 0.262999527948644 z_k1_s0107
                                 - 0.929996525274455 z_k1_s0110
                                 - 0.718000805793923 z_k1_s0111
                                 - 0.486999239427551 z_k1_s0112
                                 - 0.863998100794165 z_k1_s0113
                                 - 1.02899738048823 z_k1_s0114
                                 - 0.908999074708308 z_k1_s0115
                                 - 0.417999248056091 z_k1_s0116
                                 - 1.08199952885171 z_k1_s0117
                                 - 0.117999873574296 z_k1_s0120
                                 - 0.174999458852133 z_k1_s0121
                                 - 0.115999809864641 z_k1_s0122
                                 - 0.081999877999128 z_k1_s0123
                                 - 0.256999336851261 z_k1_s0124
                                 - 0.0999998757867122 z_k1_s0125
                                 - 0.187999369315494 z_k1_s0126
                                 - 0.403000102360729 z_k1_s0127
                                 - 0.319999905984338 z_k1_s0130
                                 - 0.109999967682116 z_k1_s0131
                                 - 0.228999856320034 z_k1_s0132
                                 - 0.105000045551119 z_k1_s0133
                                 - 0.0979999712077036 z_k1_s0134
                                 - 0.353999437595126 z_k1_s0135
                                 - 0.177999794903772 z_k1_s0136
                                 - 0.133999960630942 z_k1_s0137
                                 - 0.20899974944212 z_k1_s0140
                                 - 0.336000000485386 z_k1_s0141
                                 - 0.422999733566761 z_k1_s0142
                                 - 0.473999952131231 z_k1_s0143
                                 - 0.145999783338567 z_k1_s0144
                                 - 0.703999871540973 z_k1_s0145
                                 - 0.235999657229455 z_k1_s0146
                                 - 0.190999720284011 z_k1_s0147
                                 - 0.331999841485268 z_k1_s0150
                                 - 0.087000006608499 z_k1_s0151
                                 - 0.392999962252171 z_k1_s0152
                                 - 0.0820000807620491 z_k1_s0153
                                 - 0.0830000357713328 z_k1_s0154
                                 - 0.292999937322858 z_k1_s0155
                                 - 0.140999989910322 z_k1_s0156
                                 - 0.0970000091014302 z_k1_s0157
                                 - 0.724997350369827 z_k1_s0160
                                 - 0.431997675188298 z_k1_s0161
                                 - 0.720997407923589 z_k1_s0162
                                 - 0.388999461189619 z_k1_s0163
                                 - 0.305999488131802 z_k1_s0164
                                 - 0.46300112014124 z_k1_s0165
                                 - 0.664998213676257 z_k1_s0166
                                 - 0.189999600793062 z_k1_s0167
                                 - 0.622001467316481 z_k1_s0170
                                 - 0.200000588885401 z_k1_s0171
                                 - 0.725001895158845 z_k1_s0172
                                 - 0.279000907733399 z_k1_s0173
                                 - 0.268000738002281 z_k1_s0174
                                 - 0.448001395759534 z_k1_s0175
                                 - 0.257000887672142 z_k1_s0176
                                 - 0.394999997235092 z_k1_s0177
                                 - 0.617002606004597 z_k1_s1000
                                 - 0.269000764159055 z_k1_s1001
                                 - 0.928002301449095 z_k1_s1002
                                 - 0.276000667606806 z_k1_s1003
                                 - 0.79399939555321 z_k1_s1004
                                 - 1.06200031333432 z_k1_s1005
                                 - 0.334000566746842 z_k1_s1006
                                 - 0.614002577469694 z_k1_s1007
                                 - 0.384999510724245 z_k1_s1010
                                 - 0.384999510724245 z_k1_s1011
                                 - 0.377998974383466 z_k1_s1012
                                 - 0.782000892890416 z_k1_s1013
                                 - 0.342999624681638 z_k1_s1014
                                 - 0.166999943550613 z_k1_s1015
                                 - 0.727999135405883 z_k1_s1016
                                 - 0.164000142090545 z_k1_s1017
                                 - 0.356999937630234 z_k1_s1020
                                 - 0.175999807193113 z_k1_s1021
                                 - 0.116999960112436 z_k1_s1022
                                 - 0.126999945400683 z_k1_s1023
                                 - 0.403999670125062 z_k1_s1024
                                 - 0.111999967468313 z_k1_s1025
                                 - 0.166999225353901 z_k1_s1026
                                 - 0.192999253223318 z_k1_s1027
                                 - 0.448999053475271 z_k1_s1030
                                 - 1.15899968213755 z_k1_s1031
                                 - 0.455998949820068 z_k1_s1032
                                 - 1.05799967025162 z_k1_s1033
                                 - 0.337999058582774 z_k1_s1034
                                 - 0.523998926024525 z_k1_s1035
                                 - 0.260999281192006 z_k1_s1036
                                 - 0.342999050086058 z_k1_s1037
                                 - 0.576000545412935 z_k1_s1040
                                 - 1.58700326032256 z_k1_s1041
                                 - 1.93600014282764 z_k1_s1042
                                 - 0.930998182738801 z_k1_s1043
                                 - 0.999999355319232 z_k1_s1044
                                 - 0.73000095192468 z_k1_s1045
                                 - 0.519000882474085 z_k1_s1046
                                 - 0.624001051872878 z_k1_s1047
                                 - 0.344000301808443 z_k1_s1050
                                 - 0.424999493122811 z_k1_s1051
                                 - 0.329001110899118 z_k1_s1052
                                 - 0.240000900564808 z_k1_s1053
                                 - 0.166000276740268 z_k1_s1054
                                 - 0.131000120682873 z_k1_s1055
                                 - 0.181000258850485 z_k1_s1056
                                 - 0.162000162830743 z_k1_s1057
                                 - 0.125000034582043 z_k1_s1060
                                 - 0.181000042446799 z_k1_s1061
                                 - 0.372000392780158 z_k1_s1062
                                 - 0.22100032049305 z_k1_s1063
                                 - 0.278000549846459 z_k1_s1064
                                 - 0.0849999472357904 z_k1_s1065
                                 - 0.323999296324666 z_k1_s1066
                                 - 0.320999585358693 z_k1_s1067
                                 - 0.756001413862398 z_k1_s1070
                                 - 0.562001063523654 z_k1_s1071
                                 - 1.48999985429829 z_k1_s1072
                                 - 1.87200574699612 z_k1_s1073
                                 - 0.902003995302399 z_k1_s1074
                                 - 0.737001347134326 z_k1_s1075
                                 - 0.851001747502758 z_k1_s1076
                                 - 0.691001061460905 z_k1_s1077
                                 - 0.710000542972521 z_k1_s1100
                                 - 0.946999364389728 z_k1_s1101
                                 - 0.71799835621385 z_k1_s1102
                                 - 0.646000641037011 z_k1_s1103
                                 - 0.369000141309623 z_k1_s1104
                                 - 0.25300010160145 z_k1_s1105
                                 - 0.324000155898702 z_k1_s1106
                                 - 0.382000175752789 z_k1_s1107
                                 - 0.16500048171039 z_k1_s1110
                                 - 0.121000379707704 z_k1_s1111
                                 - 0.390002076482097 z_k1_s1112
                                 - 0.447000594362244 z_k1_s1113
                                 - 0.240001277835136 z_k1_s1114
                                 - 0.214000874868819 z_k1_s1115
                                 - 0.236001521072062 z_k1_s1116
                                 - 0.171000513656269 z_k1_s1117
                                 - 0.176999829514403 z_k1_s1120
                                 - 0.123999816563824 z_k1_s1121
                                 - 0.269999739937225 z_k1_s1122
                                 - 0.145999923373401 z_k1_s1123
                                 - 0.203999803508125 z_k1_s1124
                                 - 0.134999869968612 z_k1_s1125
                                 - 0.430999264863063 z_k1_s1126
                                 - 0.432999902936056 z_k1_s1127
                                 - 0.11799993941331 z_k1_s1130
                                 - 0.146000481099257 z_k1_s1131
                                 - 0.42399972690548 z_k1_s1132
                                 - 0.10099990233599 z_k1_s1133
                                 - 0.0909999692551723 z_k1_s1134
                                 - 0.139000251402815 z_k1_s1135
                                 - 0.114999984629053 z_k1_s1136
                                 - 0.122999905953719 z_k1_s1137
                                 - 0.207999457893545 z_k1_s1140
                                 - 0.185999507648091 z_k1_s1141
                                 - 0.306998182400365 z_k1_s1142
                                 - 0.232999616453824 z_k1_s1143
                                 - 0.405998221404343 z_k1_s1144
                                 - 0.823997801866846 z_k1_s1145
                                 - 0.195999676231975 z_k1_s1146
                                 - 0.268998698539104 z_k1_s1147
                                 - 1.02799982174363 z_k1_s1150
                                 - 0.358000436827195 z_k1_s1151
                                 - 0.658000524560308 z_k1_s1152
                                 - 0.488999227710016 z_k1_s1153
                                 - 0.495000781990303 z_k1_s1154
                                 - 0.224000096403423 z_k1_s1155
                                 - 0.224000096403423 z_k1_s1156
                                 - 0.386000352327522 z_k1_s1157
                                 - 0.198999735636325 z_k1_s1160
                                 - 0.167999672840864 z_k1_s1161
                                 - 0.231999662780078 z_k1_s1162
                                 - 0.342000382324551 z_k1_s1163
                                 - 0.488999554711079 z_k1_s1164
                                 - 0.272000318141143 z_k1_s1165
                                 - 0.526999864524537 z_k1_s1166
                                 - 0.176999784207156 z_k1_s1167
                                 - 0.0679999665388575 z_k1_s1170
                                 - 0.122999778612933 z_k1_s1171
                                 - 0.129000078642833 z_k1_s1172
                                 - 0.114000390568087 z_k1_s1173
                                 - 0.153000216762431 z_k1_s1174
                                 - 0.142000020707615 z_k1_s1175
                                 - 0.138999870692664 z_k1_s1176
                                 - 0.19399978896674 z_k1_s1177 >= 0
 z_k1_s0000_constraint:          var_1 + z_k1_s0000 + 1.26259 w_k1_ADM_s000
                                 + 1.2693 w_k1_AXP_s000 + 1.53638 w_k1_DIS_s000
                                 + 1.37894 w_k1_FDX_s000
                                 + 1.15347 w_k1_IBM_s000
                                 + 1.17893 w_k1_INTC_s000 + 1.4902 w_k1_KO_s000
                                 + 1.24463 w_k1_MAT_s000
                                 + 1.35033 w_k1_MSFT_s000
                                 + 1.35094 w_k1_PG_s000 >= 1
 z_k1_s0001_constraint:          var_1 + z_k1_s0001 + 0.900479 w_k1_ADM_s000
                                 + 0.668403 w_k1_AXP_s000
                                 + 0.655874 w_k1_DIS_s000
                                 + 0.698369 w_k1_FDX_s000
                                 + 0.780935 w_k1_IBM_s000
                                 + 0.681213 w_k1_INTC_s000
                                 + 0.829964 w_k1_KO_s000
                                 + 0.724018 w_k1_MAT_s000
                                 + 0.778626 w_k1_MSFT_s000
                                 + 0.807428 w_k1_PG_s000 >= 1
 z_k1_s0002_constraint:          var_1 + z_k1_s0002 + 1.08513 w_k1_ADM_s000
                                 + 1.05394 w_k1_AXP_s000
                                 + 0.954467 w_k1_DIS_s000
                                 + 0.963426 w_k1_FDX_s000
                                 + 1.01676 w_k1_IBM_s000
                                 + 1.03647 w_k1_INTC_s000
                                 + 0.987451 w_k1_KO_s000
                                 + 1.00986 w_k1_MAT_s000
                                 + 1.13863 w_k1_MSFT_s000
                                 + 1.00091 w_k1_PG_s000 >= 1
 z_k1_s0003_constraint:          var_1 + z_k1_s0003 + 0.753586 w_k1_ADM_s000
                                 + 0.817842 w_k1_AXP_s000
                                 + 0.837105 w_k1_DIS_s000
                                 + 0.832836 w_k1_FDX_s000
                                 + 1.04123 w_k1_IBM_s000
                                 + 0.855523 w_k1_INTC_s000
                                 + 0.827971 w_k1_KO_s000
                                 + 0.923352 w_k1_MAT_s000
                                 + 0.807105 w_k1_MSFT_s000
                                 + 0.889134 w_k1_PG_s000 >= 1
 z_k1_s0004_constraint:          var_1 + z_k1_s0004 + 1.0615 w_k1_ADM_s000
                                 + 1.1347 w_k1_AXP_s000 + 1.21333 w_k1_DIS_s000
                                 + 1.06214 w_k1_FDX_s000
                                 + 1.37637 w_k1_IBM_s000
                                 + 1.10219 w_k1_INTC_s000
                                 + 1.38325 w_k1_KO_s000 + 1.17495 w_k1_MAT_s000
                                 + 1.0911 w_k1_MSFT_s000 + 1.18806 w_k1_PG_s000
                                 >= 1
 z_k1_s0005_constraint:          var_1 + z_k1_s0005 + 1.02691 w_k1_ADM_s000
                                 + 0.785362 w_k1_AXP_s000
                                 + 1.1454 w_k1_DIS_s000
                                 + 0.921155 w_k1_FDX_s000
                                 + 0.94985 w_k1_IBM_s000
                                 + 0.755262 w_k1_INTC_s000
                                 + 0.997114 w_k1_KO_s000
                                 + 0.868056 w_k1_MAT_s000
                                 + 0.986387 w_k1_MSFT_s000
                                 + 0.948862 w_k1_PG_s000 >= 1
 z_k1_s0006_constraint:          var_1 + z_k1_s0006 + 1.39547 w_k1_ADM_s000
                                 + 1.6743 w_k1_AXP_s000 + 1.43896 w_k1_DIS_s000
                                 + 1.14537 w_k1_FDX_s000 + 1.6044 w_k1_IBM_s000
                                 + 0.871985 w_k1_INTC_s000
                                 + 1.43363 w_k1_KO_s000 + 2.0457 w_k1_MAT_s000
                                 + 1.19981 w_k1_MSFT_s000
                                 + 1.23911 w_k1_PG_s000 >= 1
 z_k1_s0007_constraint:          var_1 + z_k1_s0007 + 1.20442 w_k1_ADM_s000
                                 + 1.16118 w_k1_AXP_s000
                                 + 1.10399 w_k1_DIS_s000
                                 + 1.19399 w_k1_FDX_s000
                                 + 0.81328 w_k1_IBM_s000
                                 + 1.06077 w_k1_INTC_s000
                                 + 0.961947 w_k1_KO_s000
                                 + 0.890308 w_k1_MAT_s000
                                 + 1.18706 w_k1_MSFT_s000
                                 + 1.08232 w_k1_PG_s000 >= 1
 z_k1_s0010_constraint:          var_1 + z_k1_s0010 + 0.732154 w_k1_ADM_s001
                                 + 0.854336 w_k1_AXP_s001
                                 + 0.921288 w_k1_DIS_s001
                                 + 0.893123 w_k1_FDX_s001
                                 + 1.15265 w_k1_IBM_s001
                                 + 0.889662 w_k1_INTC_s001
                                 + 0.949031 w_k1_KO_s001
                                 + 1.18341 w_k1_MAT_s001
                                 + 0.843074 w_k1_MSFT_s001
                                 + 1.03949 w_k1_PG_s001 >= 1
 z_k1_s0011_constraint:          var_1 + z_k1_s0011 + 0.817022 w_k1_ADM_s001
                                 + 0.960972 w_k1_AXP_s001
                                 + 1.33086 w_k1_DIS_s001
                                 + 1.08678 w_k1_FDX_s001
                                 + 0.995557 w_k1_IBM_s001
                                 + 1.1189 w_k1_INTC_s001 + 1.16055 w_k1_KO_s001
                                 + 0.848686 w_k1_MAT_s001
                                 + 1.03282 w_k1_MSFT_s001 + 1.0194 w_k1_PG_s001
                                 >= 1
 z_k1_s0012_constraint:          var_1 + z_k1_s0012 + 1.14419 w_k1_ADM_s001
                                 + 1.50013 w_k1_AXP_s001 + 1.2649 w_k1_DIS_s001
                                 + 1.31947 w_k1_FDX_s001 + 1.1958 w_k1_IBM_s001
                                 + 1.48439 w_k1_INTC_s001
                                 + 1.18934 w_k1_KO_s001 + 1.13935 w_k1_MAT_s001
                                 + 1.17168 w_k1_MSFT_s001
                                 + 1.03838 w_k1_PG_s001 >= 1
 z_k1_s0013_constraint:          var_1 + z_k1_s0013 + 1.01673 w_k1_ADM_s001
                                 + 1.18382 w_k1_AXP_s001 + 1.0856 w_k1_DIS_s001
                                 + 1.21486 w_k1_FDX_s001
                                 + 0.892363 w_k1_IBM_s001
                                 + 0.99897 w_k1_INTC_s001
                                 + 0.941659 w_k1_KO_s001
                                 + 0.98352 w_k1_MAT_s001
                                 + 1.04216 w_k1_MSFT_s001
                                 + 1.05923 w_k1_PG_s001 >= 1
 z_k1_s0014_constraint:          var_1 + z_k1_s0014 + 0.826483 w_k1_ADM_s001
                                 + 0.917211 w_k1_AXP_s001
                                 + 0.726835 w_k1_DIS_s001
                                 + 0.909411 w_k1_FDX_s001
                                 + 0.817567 w_k1_IBM_s001
                                 + 0.940007 w_k1_INTC_s001
                                 + 0.731435 w_k1_KO_s001 + 0.9284 w_k1_MAT_s001
                                 + 0.723116 w_k1_MSFT_s001
                                 + 0.905971 w_k1_PG_s001 >= 1
 z_k1_s0015_constraint:          var_1 + z_k1_s0015 + 1.08807 w_k1_ADM_s001
                                 + 1.11586 w_k1_AXP_s001
                                 + 1.16148 w_k1_DIS_s001
                                 + 1.07975 w_k1_FDX_s001
                                 + 1.40453 w_k1_IBM_s001
                                 + 1.11533 w_k1_INTC_s001
                                 + 1.24161 w_k1_KO_s001 + 1.09701 w_k1_MAT_s001
                                 + 0.946638 w_k1_MSFT_s001 + 1.105 w_k1_PG_s001
                                 >= 1
 z_k1_s0016_constraint:          var_1 + z_k1_s0016 + 0.725759 w_k1_ADM_s001
                                 + 0.760525 w_k1_AXP_s001
                                 + 0.614649 w_k1_DIS_s001
                                 + 0.760202 w_k1_FDX_s001
                                 + 0.730702 w_k1_IBM_s001
                                 + 0.853119 w_k1_INTC_s001
                                 + 0.703765 w_k1_KO_s001
                                 + 0.693643 w_k1_MAT_s001
                                 + 0.708775 w_k1_MSFT_s001
                                 + 0.773476 w_k1_PG_s001 >= 1
 z_k1_s0017_constraint:          var_1 + z_k1_s0017 + 1.0023 w_k1_ADM_s001
                                 + 1.27854 w_k1_AXP_s001
                                 + 1.80011 w_k1_DIS_s001
                                 + 1.37842 w_k1_FDX_s001
                                 + 1.25437 w_k1_IBM_s001
                                 + 1.18738 w_k1_INTC_s001
                                 + 1.28026 w_k1_KO_s001 + 1.54607 w_k1_MAT_s001
                                 + 1.10406 w_k1_MSFT_s001
                                 + 1.12231 w_k1_PG_s001 >= 1
 z_k1_s0020_constraint:          var_1 + z_k1_s0020 + 1.17344 w_k1_ADM_s002
                                 + 1.60656 w_k1_AXP_s002
                                 + 1.17555 w_k1_DIS_s002
                                 + 1.17276 w_k1_FDX_s002
                                 + 1.13441 w_k1_IBM_s002
                                 + 1.12361 w_k1_INTC_s002 + 1.0806 w_k1_KO_s002
                                 + 1.71013 w_k1_MAT_s002
                                 + 1.14024 w_k1_MSFT_s002
                                 + 1.07403 w_k1_PG_s002 >= 1
 z_k1_s0021_constraint:          var_1 + z_k1_s0021 + 1.0054 w_k1_ADM_s002
                                 + 1.02726 w_k1_AXP_s002 + 1.117 w_k1_DIS_s002
                                 + 1.0558 w_k1_FDX_s002 + 1.3271 w_k1_IBM_s002
                                 + 1.15923 w_k1_INTC_s002
                                 + 1.20839 w_k1_KO_s002 + 1.65249 w_k1_MAT_s002
                                 + 1.13043 w_k1_MSFT_s002
                                 + 1.05496 w_k1_PG_s002 >= 1
 z_k1_s0022_constraint:          var_1 + z_k1_s0022 + 1.16863 w_k1_ADM_s002
                                 + 0.866881 w_k1_AXP_s002
                                 + 0.931915 w_k1_DIS_s002
                                 + 0.994122 w_k1_FDX_s002
                                 + 0.873556 w_k1_IBM_s002
                                 + 0.915436 w_k1_INTC_s002
                                 + 0.971319 w_k1_KO_s002
                                 + 1.07728 w_k1_MAT_s002
                                 + 1.03925 w_k1_MSFT_s002
                                 + 0.897732 w_k1_PG_s002 >= 1
 z_k1_s0023_constraint:          var_1 + z_k1_s0023 + 1.26704 w_k1_ADM_s002
                                 + 1.49888 w_k1_AXP_s002
                                 + 1.27174 w_k1_DIS_s002
                                 + 1.33206 w_k1_FDX_s002
                                 + 0.944558 w_k1_IBM_s002
                                 + 1.00482 w_k1_INTC_s002
                                 + 1.06541 w_k1_KO_s002 + 1.1364 w_k1_MAT_s002
                                 + 1.04755 w_k1_MSFT_s002
                                 + 1.07341 w_k1_PG_s002 >= 1
 z_k1_s0024_constraint:          var_1 + z_k1_s0024 + 1.03153 w_k1_ADM_s002
                                 + 1.07926 w_k1_AXP_s002
                                 + 0.644147 w_k1_DIS_s002
                                 + 0.953757 w_k1_FDX_s002
                                 + 0.69647 w_k1_IBM_s002
                                 + 0.853272 w_k1_INTC_s002
                                 + 0.705846 w_k1_KO_s002
                                 + 0.985957 w_k1_MAT_s002
                                 + 0.863917 w_k1_MSFT_s002
                                 + 0.812741 w_k1_PG_s002 >= 1
 z_k1_s0025_constraint:          var_1 + z_k1_s0025 + 1.41358 w_k1_ADM_s002
                                 + 1.17469 w_k1_AXP_s002
                                 + 1.30086 w_k1_DIS_s002
                                 + 1.30443 w_k1_FDX_s002
                                 + 0.896505 w_k1_IBM_s002
                                 + 1.23491 w_k1_INTC_s002
                                 + 1.08233 w_k1_KO_s002 + 1.15037 w_k1_MAT_s002
                                 + 1.2455 w_k1_MSFT_s002 + 1.06627 w_k1_PG_s002
                                 >= 1
 z_k1_s0026_constraint:          var_1 + z_k1_s0026 + 0.993807 w_k1_ADM_s002
                                 + 0.88392 w_k1_AXP_s002
                                 + 0.716989 w_k1_DIS_s002
                                 + 0.885294 w_k1_FDX_s002
                                 + 0.595964 w_k1_IBM_s002
                                 + 1.015 w_k1_INTC_s002 + 0.718449 w_k1_KO_s002
                                 + 0.624952 w_k1_MAT_s002
                                 + 0.897966 w_k1_MSFT_s002
                                 + 0.842643 w_k1_PG_s002 >= 1
 z_k1_s0027_constraint:          var_1 + z_k1_s0027 + 0.825724 w_k1_ADM_s002
                                 + 0.914724 w_k1_AXP_s002
                                 + 0.697102 w_k1_DIS_s002
                                 + 0.785678 w_k1_FDX_s002
                                 + 1.0368 w_k1_IBM_s002
                                 + 0.921012 w_k1_INTC_s002
                                 + 0.857921 w_k1_KO_s002
                                 + 1.15706 w_k1_MAT_s002
                                 + 0.840337 w_k1_MSFT_s002
                                 + 0.850207 w_k1_PG_s002 >= 1
 z_k1_s0030_constraint:          var_1 + z_k1_s0030 + 1.08159 w_k1_ADM_s003
                                 + 0.915125 w_k1_AXP_s003
                                 + 1.11208 w_k1_DIS_s003
                                 + 1.04582 w_k1_FDX_s003
                                 + 1.00025 w_k1_IBM_s003
                                 + 0.849919 w_k1_INTC_s003
                                 + 1.01272 w_k1_KO_s003
                                 + 0.961151 w_k1_MAT_s003
                                 + 0.936988 w_k1_MSFT_s003
                                 + 0.923006 w_k1_PG_s003 >= 1
 z_k1_s0031_constraint:          var_1 + z_k1_s0031 + 0.750455 w_k1_ADM_s003
                                 + 0.802319 w_k1_AXP_s003
                                 + 0.699574 w_k1_DIS_s003
                                 + 0.757855 w_k1_FDX_s003
                                 + 0.98016 w_k1_IBM_s003
                                 + 0.797303 w_k1_INTC_s003
                                 + 0.768478 w_k1_KO_s003
                                 + 0.876637 w_k1_MAT_s003
                                 + 0.781184 w_k1_MSFT_s003
                                 + 0.803097 w_k1_PG_s003 >= 1
 z_k1_s0032_constraint:          var_1 + z_k1_s0032 + 1.33835 w_k1_ADM_s003
                                 + 1.59283 w_k1_AXP_s003
                                 + 1.65584 w_k1_DIS_s003
                                 + 1.43375 w_k1_FDX_s003 + 1.1753 w_k1_IBM_s003
                                 + 1.21511 w_k1_INTC_s003
                                 + 1.21159 w_k1_KO_s003 + 1.44426 w_k1_MAT_s003
                                 + 1.44004 w_k1_MSFT_s003
                                 + 1.06528 w_k1_PG_s003 >= 1
 z_k1_s0033_constraint:          var_1 + z_k1_s0033 + 1.06756 w_k1_ADM_s003
                                 + 1.11474 w_k1_AXP_s003
                                 + 1.34064 w_k1_DIS_s003
                                 + 0.943034 w_k1_FDX_s003
                                 + 1.62343 w_k1_IBM_s003
                                 + 0.830714 w_k1_INTC_s003
                                 + 1.36896 w_k1_KO_s003 + 1.93597 w_k1_MAT_s003
                                 + 1.0044 w_k1_MSFT_s003
                                 + 0.967274 w_k1_PG_s003 >= 1
 z_k1_s0034_constraint:          var_1 + z_k1_s0034 + 0.735106 w_k1_ADM_s003
                                 + 0.974874 w_k1_AXP_s003
                                 + 1.33774 w_k1_DIS_s003
                                 + 1.18234 w_k1_FDX_s003
                                 + 0.886925 w_k1_IBM_s003
                                 + 1.08824 w_k1_INTC_s003
                                 + 1.27933 w_k1_KO_s003 + 1.24628 w_k1_MAT_s003
                                 + 1.22922 w_k1_MSFT_s003
                                 + 0.91129 w_k1_PG_s003 >= 1
 z_k1_s0035_constraint:          var_1 + z_k1_s0035 + 1.15099 w_k1_ADM_s003
                                 + 1.17045 w_k1_AXP_s003
                                 + 0.891154 w_k1_DIS_s003
                                 + 1.02481 w_k1_FDX_s003
                                 + 0.833659 w_k1_IBM_s003
                                 + 0.851466 w_k1_INTC_s003
                                 + 0.892461 w_k1_KO_s003
                                 + 1.15231 w_k1_MAT_s003
                                 + 0.885138 w_k1_MSFT_s003
                                 + 0.844613 w_k1_PG_s003 >= 1
 z_k1_s0036_constraint:          var_1 + z_k1_s0036 + 0.868694 w_k1_ADM_s003
                                 + 1.03021 w_k1_AXP_s003
                                 + 1.04817 w_k1_DIS_s003
                                 + 0.937992 w_k1_FDX_s003 + 1.135 w_k1_IBM_s003
                                 + 0.859166 w_k1_INTC_s003
                                 + 1.09273 w_k1_KO_s003 + 1.19011 w_k1_MAT_s003
                                 + 0.892239 w_k1_MSFT_s003
                                 + 0.865885 w_k1_PG_s003 >= 1
 z_k1_s0037_constraint:          var_1 + z_k1_s0037 + 0.920542 w_k1_ADM_s003
                                 + 0.852472 w_k1_AXP_s003
                                 + 0.867425 w_k1_DIS_s003
                                 + 1.02924 w_k1_FDX_s003
                                 + 0.694423 w_k1_IBM_s003
                                 + 0.860595 w_k1_INTC_s003
                                 + 0.867332 w_k1_KO_s003
                                 + 0.698221 w_k1_MAT_s003
                                 + 0.956218 w_k1_MSFT_s003
                                 + 0.802253 w_k1_PG_s003 >= 1
 z_k1_s0040_constraint:          var_1 + z_k1_s0040 + 0.936303 w_k1_ADM_s004
                                 + 0.718077 w_k1_AXP_s004
                                 + 0.631526 w_k1_DIS_s004
                                 + 0.795065 w_k1_FDX_s004
                                 + 0.626441 w_k1_IBM_s004
                                 + 0.794366 w_k1_INTC_s004
                                 + 0.654174 w_k1_KO_s004
                                 + 0.54296 w_k1_MAT_s004
                                 + 0.781083 w_k1_MSFT_s004
                                 + 0.79543 w_k1_PG_s004 >= 1
 z_k1_s0041_constraint:          var_1 + z_k1_s0041 + 1.24215 w_k1_ADM_s004
                                 + 0.966937 w_k1_AXP_s004
                                 + 1.34004 w_k1_DIS_s004
                                 + 1.19751 w_k1_FDX_s004
                                 + 0.833966 w_k1_IBM_s004
                                 + 1.03066 w_k1_INTC_s004
                                 + 1.16059 w_k1_KO_s004
                                 + 0.785778 w_k1_MAT_s004
                                 + 1.33703 w_k1_MSFT_s004 + 1.1251 w_k1_PG_s004
                                 >= 1
 z_k1_s0042_constraint:          var_1 + z_k1_s0042 + 0.987679 w_k1_ADM_s004
                                 + 0.959546 w_k1_AXP_s004
                                 + 1.11029 w_k1_DIS_s004
                                 + 0.985662 w_k1_FDX_s004
                                 + 1.0808 w_k1_IBM_s004
                                 + 0.974831 w_k1_INTC_s004
                                 + 1.07679 w_k1_KO_s004
                                 + 0.917587 w_k1_MAT_s004
                                 + 1.05453 w_k1_MSFT_s004
                                 + 1.01748 w_k1_PG_s004 >= 1
 z_k1_s0043_constraint:          var_1 + z_k1_s0043 + 1.25696 w_k1_ADM_s004
                                 + 1.63797 w_k1_AXP_s004 + 1.6477 w_k1_DIS_s004
                                 + 1.2394 w_k1_FDX_s004 + 1.48091 w_k1_IBM_s004
                                 + 1.16595 w_k1_INTC_s004
                                 + 1.57483 w_k1_KO_s004 + 1.63651 w_k1_MAT_s004
                                 + 1.32746 w_k1_MSFT_s004
                                 + 1.30461 w_k1_PG_s004 >= 1
 z_k1_s0044_constraint:          var_1 + z_k1_s0044 + 1.08603 w_k1_ADM_s004
                                 + 0.952166 w_k1_AXP_s004
                                 + 0.776306 w_k1_DIS_s004
                                 + 0.832798 w_k1_FDX_s004
                                 + 0.984149 w_k1_IBM_s004
                                 + 0.863401 w_k1_INTC_s004
                                 + 0.852939 w_k1_KO_s004
                                 + 0.841875 w_k1_MAT_s004
                                 + 0.85179 w_k1_MSFT_s004
                                 + 0.930298 w_k1_PG_s004 >= 1
 z_k1_s0045_constraint:          var_1 + z_k1_s0045 + 0.821155 w_k1_ADM_s004
                                 + 1.01388 w_k1_AXP_s004
                                 + 0.864552 w_k1_DIS_s004
                                 + 0.783263 w_k1_FDX_s004
                                 + 1.58696 w_k1_IBM_s004
                                 + 0.80286 w_k1_INTC_s004
                                 + 1.02512 w_k1_KO_s004 + 1.42011 w_k1_MAT_s004
                                 + 0.846791 w_k1_MSFT_s004
                                 + 0.905491 w_k1_PG_s004 >= 1
 z_k1_s0046_constraint:          var_1 + z_k1_s0046 + 1.38707 w_k1_ADM_s004
                                 + 1.46003 w_k1_AXP_s004
                                 + 1.11013 w_k1_DIS_s004
                                 + 1.28734 w_k1_FDX_s004
                                 + 0.876514 w_k1_IBM_s004
                                 + 1.01713 w_k1_INTC_s004
                                 + 1.12574 w_k1_KO_s004 + 1.15209 w_k1_MAT_s004
                                 + 1.15148 w_k1_MSFT_s004
                                 + 1.23784 w_k1_PG_s004 >= 1
 z_k1_s0047_constraint:          var_1 + z_k1_s0047 + 1.08847 w_k1_ADM_s004
                                 + 1.3705 w_k1_AXP_s004 + 1.27283 w_k1_DIS_s004
                                 + 1.08974 w_k1_FDX_s004
                                 + 1.19165 w_k1_IBM_s004
                                 + 1.12077 w_k1_INTC_s004
                                 + 1.19051 w_k1_KO_s004 + 1.35859 w_k1_MAT_s004
                                 + 1.1643 w_k1_MSFT_s004 + 1.10165 w_k1_PG_s004
                                 >= 1
 z_k1_s0050_constraint:          var_1 + z_k1_s0050 + 0.953385 w_k1_ADM_s005
                                 + 1.24584 w_k1_AXP_s005
                                 + 0.937721 w_k1_DIS_s005
                                 + 1.0301 w_k1_FDX_s005 + 1.14537 w_k1_IBM_s005
                                 + 0.995685 w_k1_INTC_s005
                                 + 1.06035 w_k1_KO_s005 + 1.12272 w_k1_MAT_s005
                                 + 0.953405 w_k1_MSFT_s005
                                 + 1.02902 w_k1_PG_s005 >= 1
 z_k1_s0051_constraint:          var_1 + z_k1_s0051 + 1.50341 w_k1_ADM_s005
                                 + 1.63585 w_k1_AXP_s005
                                 + 1.21633 w_k1_DIS_s005
                                 + 1.29698 w_k1_FDX_s005
                                 + 0.932144 w_k1_IBM_s005
                                 + 0.981135 w_k1_INTC_s005
                                 + 1.11582 w_k1_KO_s005 + 0.96612 w_k1_MAT_s005
                                 + 1.07984 w_k1_MSFT_s005
                                 + 0.980368 w_k1_PG_s005 >= 1
 z_k1_s0052_constraint:          var_1 + z_k1_s0052 + 1.3697 w_k1_ADM_s005
                                 + 1.13098 w_k1_AXP_s005
                                 + 0.946404 w_k1_DIS_s005
                                 + 1.00509 w_k1_FDX_s005
                                 + 0.858285 w_k1_IBM_s005
                                 + 0.991913 w_k1_INTC_s005
                                 + 0.919624 w_k1_KO_s005
                                 + 0.722002 w_k1_MAT_s005
                                 + 0.922865 w_k1_MSFT_s005
                                 + 0.943259 w_k1_PG_s005 >= 1
 z_k1_s0053_constraint:          var_1 + z_k1_s0053 + 0.896321 w_k1_ADM_s005
                                 + 1.28844 w_k1_AXP_s005 + 1.3389 w_k1_DIS_s005
                                 + 1.03615 w_k1_FDX_s005
                                 + 1.39934 w_k1_IBM_s005
                                 + 1.07832 w_k1_INTC_s005
                                 + 1.37696 w_k1_KO_s005 + 2.07164 w_k1_MAT_s005
                                 + 1.21119 w_k1_MSFT_s005 + 1.1303 w_k1_PG_s005
                                 >= 1
 z_k1_s0054_constraint:          var_1 + z_k1_s0054 + 0.76846 w_k1_ADM_s005
                                 + 0.717422 w_k1_AXP_s005
                                 + 0.669591 w_k1_DIS_s005
                                 + 0.762328 w_k1_FDX_s005
                                 + 1.01324 w_k1_IBM_s005
                                 + 0.685584 w_k1_INTC_s005
                                 + 0.864905 w_k1_KO_s005
                                 + 1.00985 w_k1_MAT_s005
                                 + 0.800504 w_k1_MSFT_s005
                                 + 0.855069 w_k1_PG_s005 >= 1
 z_k1_s0055_constraint:          var_1 + z_k1_s0055 + 1.02952 w_k1_ADM_s005
                                 + 1.02066 w_k1_AXP_s005 + 1.157 w_k1_DIS_s005
                                 + 1.08857 w_k1_FDX_s005
                                 + 1.21519 w_k1_IBM_s005
                                 + 1.06829 w_k1_INTC_s005
                                 + 1.14169 w_k1_KO_s005 + 1.4534 w_k1_MAT_s005
                                 + 1.11959 w_k1_MSFT_s005
                                 + 0.991017 w_k1_PG_s005 >= 1
 z_k1_s0056_constraint:          var_1 + z_k1_s0056 + 0.668477 w_k1_ADM_s005
                                 + 0.873555 w_k1_AXP_s005
                                 + 0.720676 w_k1_DIS_s005
                                 + 0.952041 w_k1_FDX_s005
                                 + 0.637487 w_k1_IBM_s005
                                 + 0.845785 w_k1_INTC_s005
                                 + 0.789671 w_k1_KO_s005
                                 + 1.07458 w_k1_MAT_s005
                                 + 0.823038 w_k1_MSFT_s005
                                 + 0.843718 w_k1_PG_s005 >= 1
 z_k1_s0057_constraint:          var_1 + z_k1_s0057 + 0.947235 w_k1_ADM_s005
                                 + 0.854633 w_k1_AXP_s005
                                 + 1.05783 w_k1_DIS_s005
                                 + 0.926849 w_k1_FDX_s005
                                 + 0.851945 w_k1_IBM_s005
                                 + 1.0639 w_k1_INTC_s005 + 1.09211 w_k1_KO_s005
                                 + 0.790207 w_k1_MAT_s005
                                 + 0.936463 w_k1_MSFT_s005
                                 + 0.856834 w_k1_PG_s005 >= 1
 z_k1_s0060_constraint:          var_1 + z_k1_s0060 + 0.718172 w_k1_ADM_s006
                                 + 0.607218 w_k1_AXP_s006
                                 + 0.561137 w_k1_DIS_s006
                                 + 0.709659 w_k1_FDX_s006
                                 + 0.822668 w_k1_IBM_s006
                                 + 0.683734 w_k1_INTC_s006
                                 + 0.756627 w_k1_KO_s006
                                 + 1.11214 w_k1_MAT_s006
                                 + 0.722134 w_k1_MSFT_s006
                                 + 0.876644 w_k1_PG_s006 >= 1
 z_k1_s0061_constraint:          var_1 + z_k1_s0061 + 1.32557 w_k1_ADM_s006
                                 + 1.3293 w_k1_AXP_s006 + 1.29428 w_k1_DIS_s006
                                 + 1.12359 w_k1_FDX_s006
                                 + 1.46492 w_k1_IBM_s006
                                 + 1.04512 w_k1_INTC_s006
                                 + 1.20239 w_k1_KO_s006 + 1.90714 w_k1_MAT_s006
                                 + 1.02421 w_k1_MSFT_s006
                                 + 1.10298 w_k1_PG_s006 >= 1
 z_k1_s0062_constraint:          var_1 + z_k1_s0062 + 1.01365 w_k1_ADM_s006
                                 + 1.13345 w_k1_AXP_s006
                                 + 0.90425 w_k1_DIS_s006
                                 + 0.893485 w_k1_FDX_s006
                                 + 1.34229 w_k1_IBM_s006
                                 + 0.930721 w_k1_INTC_s006
                                 + 1.01553 w_k1_KO_s006 + 1.53401 w_k1_MAT_s006
                                 + 0.862917 w_k1_MSFT_s006
                                 + 0.91045 w_k1_PG_s006 >= 1
 z_k1_s0063_constraint:          var_1 + z_k1_s0063 + 0.899231 w_k1_ADM_s006
                                 + 0.884471 w_k1_AXP_s006
                                 + 0.866034 w_k1_DIS_s006
                                 + 0.928586 w_k1_FDX_s006
                                 + 0.993327 w_k1_IBM_s006
                                 + 0.815236 w_k1_INTC_s006
                                 + 0.865715 w_k1_KO_s006
                                 + 0.915451 w_k1_MAT_s006
                                 + 0.863818 w_k1_MSFT_s006
                                 + 0.94645 w_k1_PG_s006 >= 1
 z_k1_s0064_constraint:          var_1 + z_k1_s0064 + 1.16471 w_k1_ADM_s006
                                 + 0.952098 w_k1_AXP_s006
                                 + 0.978618 w_k1_DIS_s006
                                 + 0.924927 w_k1_FDX_s006
                                 + 0.881872 w_k1_IBM_s006
                                 + 0.792867 w_k1_INTC_s006
                                 + 1.01807 w_k1_KO_s006 + 1.26162 w_k1_MAT_s006
                                 + 0.94644 w_k1_MSFT_s006
                                 + 0.955582 w_k1_PG_s006 >= 1
 z_k1_s0065_constraint:          var_1 + z_k1_s0065 + 1.10184 w_k1_ADM_s006
                                 + 0.999328 w_k1_AXP_s006
                                 + 0.87142 w_k1_DIS_s006
                                 + 1.05884 w_k1_FDX_s006
                                 + 0.656827 w_k1_IBM_s006
                                 + 0.803224 w_k1_INTC_s006
                                 + 0.681437 w_k1_KO_s006
                                 + 1.00129 w_k1_MAT_s006
                                 + 0.820132 w_k1_MSFT_s006
                                 + 0.883937 w_k1_PG_s006 >= 1
 z_k1_s0066_constraint:          var_1 + z_k1_s0066 + 1.33567 w_k1_ADM_s006
                                 + 1.43129 w_k1_AXP_s006
                                 + 1.89304 w_k1_DIS_s006
                                 + 1.36019 w_k1_FDX_s006
                                 + 1.50519 w_k1_IBM_s006
                                 + 1.31581 w_k1_INTC_s006
                                 + 1.35026 w_k1_KO_s006 + 1.23981 w_k1_MAT_s006
                                 + 1.3298 w_k1_MSFT_s006 + 1.19382 w_k1_PG_s006
                                 >= 1
 z_k1_s0067_constraint:          var_1 + z_k1_s0067 + 1.46429 w_k1_ADM_s006
                                 + 1.05301 w_k1_AXP_s006
                                 + 1.23059 w_k1_DIS_s006 + 1.1869 w_k1_FDX_s006
                                 + 0.98326 w_k1_IBM_s006
                                 + 1.08434 w_k1_INTC_s006
                                 + 1.16486 w_k1_KO_s006 + 1.30737 w_k1_MAT_s006
                                 + 1.25707 w_k1_MSFT_s006
                                 + 1.15459 w_k1_PG_s006 >= 1
 z_k1_s0070_constraint:          var_1 + z_k1_s0070 + 1.50946 w_k1_ADM_s007
                                 + 1.17697 w_k1_AXP_s007
                                 + 0.809803 w_k1_DIS_s007
                                 + 1.02923 w_k1_FDX_s007
                                 + 1.03931 w_k1_IBM_s007
                                 + 0.820069 w_k1_INTC_s007
                                 + 0.856282 w_k1_KO_s007 + 1.5023 w_k1_MAT_s007
                                 + 0.932351 w_k1_MSFT_s007
                                 + 1.08876 w_k1_PG_s007 >= 1
 z_k1_s0071_constraint:          var_1 + z_k1_s0071 + 0.81563 w_k1_ADM_s007
                                 + 0.866896 w_k1_AXP_s007
                                 + 0.833785 w_k1_DIS_s007
                                 + 0.838597 w_k1_FDX_s007
                                 + 1.12047 w_k1_IBM_s007
                                 + 0.824633 w_k1_INTC_s007
                                 + 0.973663 w_k1_KO_s007
                                 + 1.23262 w_k1_MAT_s007
                                 + 0.909907 w_k1_MSFT_s007
                                 + 0.820199 w_k1_PG_s007 >= 1
 z_k1_s0072_constraint:          var_1 + z_k1_s0072 + 1.14938 w_k1_ADM_s007
                                 + 1.19055 w_k1_AXP_s007
                                 + 1.03925 w_k1_DIS_s007
                                 + 1.21823 w_k1_FDX_s007
                                 + 0.740992 w_k1_IBM_s007
                                 + 1.29482 w_k1_INTC_s007
                                 + 0.905155 w_k1_KO_s007
                                 + 0.903865 w_k1_MAT_s007
                                 + 1.31942 w_k1_MSFT_s007
                                 + 0.984872 w_k1_PG_s007 >= 1
 z_k1_s0073_constraint:          var_1 + z_k1_s0073 + 1.17996 w_k1_ADM_s007
                                 + 0.864011 w_k1_AXP_s007
                                 + 1.08313 w_k1_DIS_s007
                                 + 1.00337 w_k1_FDX_s007
                                 + 0.705017 w_k1_IBM_s007
                                 + 0.862798 w_k1_INTC_s007
                                 + 0.902594 w_k1_KO_s007
                                 + 0.615074 w_k1_MAT_s007
                                 + 1.09196 w_k1_MSFT_s007
                                 + 0.885216 w_k1_PG_s007 >= 1
 z_k1_s0074_constraint:          var_1 + z_k1_s0074 + 0.88812 w_k1_ADM_s007
                                 + 0.90194 w_k1_AXP_s007
                                 + 0.633959 w_k1_DIS_s007
                                 + 0.788004 w_k1_FDX_s007
                                 + 0.703198 w_k1_IBM_s007
                                 + 0.921099 w_k1_INTC_s007
                                 + 0.598559 w_k1_KO_s007
                                 + 0.737498 w_k1_MAT_s007
                                 + 0.863276 w_k1_MSFT_s007
                                 + 0.865931 w_k1_PG_s007 >= 1
 z_k1_s0075_constraint:          var_1 + z_k1_s0075 + 1.24043 w_k1_ADM_s007
                                 + 1.28064 w_k1_AXP_s007 + 1.0441 w_k1_DIS_s007
                                 + 1.05595 w_k1_FDX_s007
                                 + 1.48748 w_k1_IBM_s007
                                 + 1.02788 w_k1_INTC_s007
                                 + 1.07972 w_k1_KO_s007 + 1.73984 w_k1_MAT_s007
                                 + 1.0347 w_k1_MSFT_s007 + 0.99838 w_k1_PG_s007
                                 >= 1
 z_k1_s0076_constraint:          var_1 + z_k1_s0076 + 1.17468 w_k1_ADM_s007
                                 + 1.35698 w_k1_AXP_s007 + 1.5736 w_k1_DIS_s007
                                 + 1.24674 w_k1_FDX_s007
                                 + 1.54883 w_k1_IBM_s007
                                 + 1.4516 w_k1_INTC_s007 + 1.40361 w_k1_KO_s007
                                 + 1.55613 w_k1_MAT_s007
                                 + 1.34127 w_k1_MSFT_s007
                                 + 1.24617 w_k1_PG_s007 >= 1
 z_k1_s0077_constraint:          var_1 + z_k1_s0077 + 0.828428 w_k1_ADM_s007
                                 + 0.859896 w_k1_AXP_s007
                                 + 1.13448 w_k1_DIS_s007
                                 + 0.875228 w_k1_FDX_s007
                                 + 1.4869 w_k1_IBM_s007
                                 + 1.01812 w_k1_INTC_s007
                                 + 1.09402 w_k1_KO_s007 + 1.42036 w_k1_MAT_s007
                                 + 1.08367 w_k1_MSFT_s007
                                 + 0.936421 w_k1_PG_s007 >= 1
 z_k1_s0100_constraint:          var_1 + z_k1_s0100 + 0.832304 w_k1_ADM_s010
                                 + 0.666249 w_k1_AXP_s010
                                 + 0.740094 w_k1_DIS_s010
                                 + 0.801816 w_k1_FDX_s010
                                 + 0.79113 w_k1_IBM_s010
                                 + 0.835711 w_k1_INTC_s010
                                 + 0.825598 w_k1_KO_s010
                                 + 0.717376 w_k1_MAT_s010
                                 + 0.937428 w_k1_MSFT_s010
                                 + 0.848555 w_k1_PG_s010 >= 1
 z_k1_s0101_constraint:          var_1 + z_k1_s0101 + 1.11983 w_k1_ADM_s010
                                 + 1.14088 w_k1_AXP_s010
                                 + 1.19385 w_k1_DIS_s010
                                 + 1.08352 w_k1_FDX_s010
                                 + 1.00802 w_k1_IBM_s010
                                 + 1.04807 w_k1_INTC_s010
                                 + 1.01236 w_k1_KO_s010 + 1.43034 w_k1_MAT_s010
                                 + 1.22471 w_k1_MSFT_s010
                                 + 1.06416 w_k1_PG_s010 >= 1
 z_k1_s0102_constraint:          var_1 + z_k1_s0102 + 0.77039 w_k1_ADM_s010
                                 + 0.68932 w_k1_AXP_s010
                                 + 0.835275 w_k1_DIS_s010
                                 + 0.802652 w_k1_FDX_s010
                                 + 1.11955 w_k1_IBM_s010
                                 + 0.953428 w_k1_INTC_s010
                                 + 0.958769 w_k1_KO_s010
                                 + 1.20911 w_k1_MAT_s010
                                 + 1.0177 w_k1_MSFT_s010
                                 + 0.933598 w_k1_PG_s010 >= 1
 z_k1_s0103_constraint:          var_1 + z_k1_s0103 + 0.879254 w_k1_ADM_s010
                                 + 0.970171 w_k1_AXP_s010
                                 + 0.752367 w_k1_DIS_s010
                                 + 0.956982 w_k1_FDX_s010
                                 + 0.618404 w_k1_IBM_s010
                                 + 1.00839 w_k1_INTC_s010
                                 + 0.709406 w_k1_KO_s010
                                 + 0.83106 w_k1_MAT_s010
                                 + 0.963125 w_k1_MSFT_s010
                                 + 0.899523 w_k1_PG_s010 >= 1
 z_k1_s0104_constraint:          var_1 + z_k1_s0104 + 1.06772 w_k1_ADM_s010
                                 + 1.17747 w_k1_AXP_s010
                                 + 1.50301 w_k1_DIS_s010 + 1.1061 w_k1_FDX_s010
                                 + 1.53242 w_k1_IBM_s010
                                 + 1.06273 w_k1_INTC_s010
                                 + 1.31923 w_k1_KO_s010 + 1.65216 w_k1_MAT_s010
                                 + 1.10513 w_k1_MSFT_s010
                                 + 1.17593 w_k1_PG_s010 >= 1
 z_k1_s0105_constraint:          var_1 + z_k1_s0105 + 1.33359 w_k1_ADM_s010
                                 + 1.01006 w_k1_AXP_s010
                                 + 1.46397 w_k1_DIS_s010
                                 + 1.29401 w_k1_FDX_s010
                                 + 0.873935 w_k1_IBM_s010
                                 + 1.04217 w_k1_INTC_s010
                                 + 1.06881 w_k1_KO_s010
                                 + 0.757003 w_k1_MAT_s010
                                 + 1.39391 w_k1_MSFT_s010
                                 + 1.01974 w_k1_PG_s010 >= 1
 z_k1_s0106_constraint:          var_1 + z_k1_s0106 + 1.1221 w_k1_ADM_s010
                                 + 0.872007 w_k1_AXP_s010
                                 + 0.971946 w_k1_DIS_s010
                                 + 0.986408 w_k1_FDX_s010
                                 + 0.832908 w_k1_IBM_s010
                                 + 0.952081 w_k1_INTC_s010
                                 + 0.870883 w_k1_KO_s010
                                 + 0.787125 w_k1_MAT_s010
                                 + 1.20357 w_k1_MSFT_s010
                                 + 0.919232 w_k1_PG_s010 >= 1
 z_k1_s0107_constraint:          var_1 + z_k1_s0107 + 1.49852 w_k1_ADM_s010
                                 + 1.56358 w_k1_AXP_s010
                                 + 1.94051 w_k1_DIS_s010
                                 + 1.48642 w_k1_FDX_s010
                                 + 1.16487 w_k1_IBM_s010
                                 + 1.48428 w_k1_INTC_s010
                                 + 1.28265 w_k1_KO_s010 + 1.05815 w_k1_MAT_s010
                                 + 1.54528 w_k1_MSFT_s010
                                 + 1.30552 w_k1_PG_s010 >= 1
 z_k1_s0110_constraint:          var_1 + z_k1_s0110 + 1.32696 w_k1_ADM_s011
                                 + 1.29005 w_k1_AXP_s011
                                 + 1.72588 w_k1_DIS_s011
                                 + 1.37961 w_k1_FDX_s011
                                 + 1.16576 w_k1_IBM_s011
                                 + 1.2308 w_k1_INTC_s011 + 1.31359 w_k1_KO_s011
                                 + 1.19686 w_k1_MAT_s011
                                 + 1.2539 w_k1_MSFT_s011 + 1.27369 w_k1_PG_s011
                                 >= 1
 z_k1_s0111_constraint:          var_1 + z_k1_s0111 + 1.04633 w_k1_ADM_s011
                                 + 1.51933 w_k1_AXP_s011
                                 + 1.34917 w_k1_DIS_s011
                                 + 1.08777 w_k1_FDX_s011
                                 + 1.74815 w_k1_IBM_s011
                                 + 1.07719 w_k1_INTC_s011
                                 + 1.41528 w_k1_KO_s011 + 1.83601 w_k1_MAT_s011
                                 + 1.06803 w_k1_MSFT_s011
                                 + 1.20897 w_k1_PG_s011 >= 1
 z_k1_s0112_constraint:          var_1 + z_k1_s0112 + 0.958426 w_k1_ADM_s011
                                 + 1.01871 w_k1_AXP_s011 + 1.1007 w_k1_DIS_s011
                                 + 0.946062 w_k1_FDX_s011
                                 + 1.22674 w_k1_IBM_s011
                                 + 0.937476 w_k1_INTC_s011 + 1.129 w_k1_KO_s011
                                 + 1.01278 w_k1_MAT_s011
                                 + 0.907774 w_k1_MSFT_s011
                                 + 0.956355 w_k1_PG_s011 >= 1
 z_k1_s0113_constraint:          var_1 + z_k1_s0113 + 1.20856 w_k1_ADM_s011
                                 + 1.14664 w_k1_AXP_s011
                                 + 1.07955 w_k1_DIS_s011
                                 + 1.07993 w_k1_FDX_s011
                                 + 1.11401 w_k1_IBM_s011
                                 + 0.969826 w_k1_INTC_s011
                                 + 1.02659 w_k1_KO_s011 + 1.12614 w_k1_MAT_s011
                                 + 1.0042 w_k1_MSFT_s011
                                 + 0.958083 w_k1_PG_s011 >= 1
 z_k1_s0114_constraint:          var_1 + z_k1_s0114 + 0.999387 w_k1_ADM_s011
                                 + 0.938125 w_k1_AXP_s011
                                 + 1.00881 w_k1_DIS_s011
                                 + 1.07776 w_k1_FDX_s011
                                 + 0.817746 w_k1_IBM_s011
                                 + 0.996965 w_k1_INTC_s011
                                 + 0.930155 w_k1_KO_s011
                                 + 0.810627 w_k1_MAT_s011
                                 + 1.11744 w_k1_MSFT_s011
                                 + 1.03332 w_k1_PG_s011 >= 1
 z_k1_s0115_constraint:          var_1 + z_k1_s0115 + 0.948038 w_k1_ADM_s011
                                 + 0.783855 w_k1_AXP_s011
                                 + 0.872422 w_k1_DIS_s011
                                 + 0.974443 w_k1_FDX_s011
                                 + 0.523004 w_k1_IBM_s011
                                 + 0.971341 w_k1_INTC_s011
                                 + 0.76118 w_k1_KO_s011
                                 + 0.449575 w_k1_MAT_s011
                                 + 1.00281 w_k1_MSFT_s011
                                 + 0.880089 w_k1_PG_s011 >= 1
 z_k1_s0116_constraint:          var_1 + z_k1_s0116 + 1.12748 w_k1_ADM_s011
                                 + 1.69943 w_k1_AXP_s011
                                 + 1.07616 w_k1_DIS_s011
                                 + 1.14874 w_k1_FDX_s011
                                 + 0.92596 w_k1_IBM_s011
                                 + 1.02085 w_k1_INTC_s011
                                 + 1.00673 w_k1_KO_s011 + 1.37032 w_k1_MAT_s011
                                 + 1.08099 w_k1_MSFT_s011
                                 + 1.06128 w_k1_PG_s011 >= 1
 z_k1_s0117_constraint:          var_1 + z_k1_s0117 + 0.956957 w_k1_ADM_s011
                                 + 0.73886 w_k1_AXP_s011
                                 + 0.597156 w_k1_DIS_s011
                                 + 0.764769 w_k1_FDX_s011
                                 + 0.721325 w_k1_IBM_s011
                                 + 0.722908 w_k1_INTC_s011
                                 + 0.713255 w_k1_KO_s011
                                 + 0.823537 w_k1_MAT_s011
                                 + 0.848293 w_k1_MSFT_s011
                                 + 0.81894 w_k1_PG_s011 >= 1
 z_k1_s0120_constraint:          var_1 + z_k1_s0120 + 0.904887 w_k1_ADM_s012
                                 + 0.677418 w_k1_AXP_s012
                                 + 0.643688 w_k1_DIS_s012
                                 + 0.852114 w_k1_FDX_s012
                                 + 0.633004 w_k1_IBM_s012
                                 + 0.952071 w_k1_INTC_s012
                                 + 0.704798 w_k1_KO_s012
                                 + 0.683169 w_k1_MAT_s012
                                 + 0.766101 w_k1_MSFT_s012
                                 + 0.74313 w_k1_PG_s012 >= 1
 z_k1_s0121_constraint:          var_1 + z_k1_s0121 + 1.13873 w_k1_ADM_s012
                                 + 0.798496 w_k1_AXP_s012 + 1.174 w_k1_DIS_s012
                                 + 1.06416 w_k1_FDX_s012
                                 + 0.763683 w_k1_IBM_s012
                                 + 0.838682 w_k1_INTC_s012
                                 + 1.04021 w_k1_KO_s012
                                 + 0.896991 w_k1_MAT_s012
                                 + 1.08654 w_k1_MSFT_s012
                                 + 0.846771 w_k1_PG_s012 >= 1
 z_k1_s0122_constraint:          var_1 + z_k1_s0122 + 1.12354 w_k1_ADM_s012
                                 + 1.1816 w_k1_AXP_s012
                                 + 0.938438 w_k1_DIS_s012
                                 + 1.03919 w_k1_FDX_s012
                                 + 0.79259 w_k1_IBM_s012
                                 + 0.795328 w_k1_INTC_s012
                                 + 0.876029 w_k1_KO_s012
                                 + 1.18003 w_k1_MAT_s012
                                 + 0.920225 w_k1_MSFT_s012
                                 + 0.964717 w_k1_PG_s012 >= 1
 z_k1_s0123_constraint:          var_1 + z_k1_s0123 + 1.60852 w_k1_ADM_s012
                                 + 1.58085 w_k1_AXP_s012
                                 + 1.82559 w_k1_DIS_s012 + 1.3023 w_k1_FDX_s012
                                 + 1.75855 w_k1_IBM_s012
                                 + 1.04754 w_k1_INTC_s012
                                 + 1.81601 w_k1_KO_s012 + 2.17113 w_k1_MAT_s012
                                 + 1.25255 w_k1_MSFT_s012
                                 + 1.23301 w_k1_PG_s012 >= 1
 z_k1_s0124_constraint:          var_1 + z_k1_s0124 + 1.11037 w_k1_ADM_s012
                                 + 0.806966 w_k1_AXP_s012
                                 + 0.853708 w_k1_DIS_s012
                                 + 0.90756 w_k1_FDX_s012
                                 + 0.909036 w_k1_IBM_s012
                                 + 0.76846 w_k1_INTC_s012
                                 + 0.812685 w_k1_KO_s012
                                 + 0.929649 w_k1_MAT_s012
                                 + 0.821013 w_k1_MSFT_s012
                                 + 0.884953 w_k1_PG_s012 >= 1
 z_k1_s0125_constraint:          var_1 + z_k1_s0125 + 1.60131 w_k1_ADM_s012
                                 + 1.24838 w_k1_AXP_s012
                                 + 1.86164 w_k1_DIS_s012
                                 + 1.33299 w_k1_FDX_s012
                                 + 1.02227 w_k1_IBM_s012
                                 + 1.24071 w_k1_INTC_s012
                                 + 1.54683 w_k1_KO_s012
                                 + 0.906328 w_k1_MAT_s012
                                 + 1.5914 w_k1_MSFT_s012 + 1.14287 w_k1_PG_s012
                                 >= 1
 z_k1_s0126_constraint:          var_1 + z_k1_s0126 + 1.05757 w_k1_ADM_s012
                                 + 1.00817 w_k1_AXP_s012
                                 + 0.807873 w_k1_DIS_s012
                                 + 0.806106 w_k1_FDX_s012
                                 + 1.2117 w_k1_IBM_s012
                                 + 0.850205 w_k1_INTC_s012
                                 + 0.968283 w_k1_KO_s012
                                 + 1.34621 w_k1_MAT_s012
                                 + 0.761846 w_k1_MSFT_s012
                                 + 0.940148 w_k1_PG_s012 >= 1
 z_k1_s0127_constraint:          var_1 + z_k1_s0127 + 1.36051 w_k1_ADM_s012
                                 + 1.21191 w_k1_AXP_s012
                                 + 1.26629 w_k1_DIS_s012
                                 + 1.14997 w_k1_FDX_s012
                                 + 1.11861 w_k1_IBM_s012
                                 + 1.18441 w_k1_INTC_s012 + 1.0922 w_k1_KO_s012
                                 + 1.02802 w_k1_MAT_s012
                                 + 1.03192 w_k1_MSFT_s012
                                 + 0.915081 w_k1_PG_s012 >= 1
 z_k1_s0130_constraint:          var_1 + z_k1_s0130 + 1.08954 w_k1_ADM_s013
                                 + 0.902374 w_k1_AXP_s013
                                 + 0.910986 w_k1_DIS_s013
                                 + 0.944476 w_k1_FDX_s013
                                 + 1.25324 w_k1_IBM_s013
                                 + 1.09976 w_k1_INTC_s013
                                 + 1.20495 w_k1_KO_s013 + 1.34277 w_k1_MAT_s013
                                 + 0.932637 w_k1_MSFT_s013
                                 + 1.01662 w_k1_PG_s013 >= 1
 z_k1_s0131_constraint:          var_1 + z_k1_s0131 + 1.20137 w_k1_ADM_s013
                                 + 1.11528 w_k1_AXP_s013
                                 + 1.09213 w_k1_DIS_s013
                                 + 1.24815 w_k1_FDX_s013
                                 + 0.88312 w_k1_IBM_s013
                                 + 1.24035 w_k1_INTC_s013
                                 + 1.03824 w_k1_KO_s013
                                 + 0.828875 w_k1_MAT_s013
                                 + 1.18927 w_k1_MSFT_s013
                                 + 1.06087 w_k1_PG_s013 >= 1
 z_k1_s0132_constraint:          var_1 + z_k1_s0132 + 1.36482 w_k1_ADM_s013
                                 + 1.2243 w_k1_AXP_s013 + 1.50086 w_k1_DIS_s013
                                 + 1.2258 w_k1_FDX_s013 + 1.4762 w_k1_IBM_s013
                                 + 1.31544 w_k1_INTC_s013
                                 + 1.35263 w_k1_KO_s013 + 1.52973 w_k1_MAT_s013
                                 + 1.37964 w_k1_MSFT_s013
                                 + 1.20645 w_k1_PG_s013 >= 1
 z_k1_s0133_constraint:          var_1 + z_k1_s0133 + 1.03844 w_k1_ADM_s013
                                 + 0.834529 w_k1_AXP_s013
                                 + 1.60691 w_k1_DIS_s013
                                 + 1.11438 w_k1_FDX_s013
                                 + 1.16025 w_k1_IBM_s013
                                 + 1.22551 w_k1_INTC_s013 + 1.3261 w_k1_KO_s013
                                 + 1.00011 w_k1_MAT_s013
                                 + 1.3758 w_k1_MSFT_s013 + 1.10154 w_k1_PG_s013
                                 >= 1
 z_k1_s0134_constraint:          var_1 + z_k1_s0134 + 0.776113 w_k1_ADM_s013
                                 + 0.651559 w_k1_AXP_s013
                                 + 0.547513 w_k1_DIS_s013
                                 + 0.769592 w_k1_FDX_s013
                                 + 0.627981 w_k1_IBM_s013
                                 + 0.938345 w_k1_INTC_s013
                                 + 0.676915 w_k1_KO_s013
                                 + 0.590969 w_k1_MAT_s013
                                 + 0.835194 w_k1_MSFT_s013
                                 + 0.848592 w_k1_PG_s013 >= 1
 z_k1_s0135_constraint:          var_1 + z_k1_s0135 + 0.98951 w_k1_ADM_s013
                                 + 0.847372 w_k1_AXP_s013
                                 + 1.0086 w_k1_DIS_s013 + 1.01056 w_k1_FDX_s013
                                 + 0.802197 w_k1_IBM_s013
                                 + 1.04549 w_k1_INTC_s013
                                 + 0.886267 w_k1_KO_s013
                                 + 0.828427 w_k1_MAT_s013
                                 + 1.03575 w_k1_MSFT_s013
                                 + 0.883279 w_k1_PG_s013 >= 1
 z_k1_s0136_constraint:          var_1 + z_k1_s0136 + 0.894823 w_k1_ADM_s013
                                 + 0.572961 w_k1_AXP_s013
                                 + 0.755749 w_k1_DIS_s013
                                 + 0.727104 w_k1_FDX_s013
                                 + 1.00556 w_k1_IBM_s013
                                 + 1.01302 w_k1_INTC_s013
                                 + 0.956674 w_k1_KO_s013
                                 + 0.699341 w_k1_MAT_s013
                                 + 0.878723 w_k1_MSFT_s013
                                 + 0.867264 w_k1_PG_s013 >= 1
 z_k1_s0137_constraint:          var_1 + z_k1_s0137 + 1.08062 w_k1_ADM_s013
                                 + 0.835246 w_k1_AXP_s013
                                 + 0.72732 w_k1_DIS_s013
                                 + 0.971071 w_k1_FDX_s013
                                 + 0.725089 w_k1_IBM_s013
                                 + 0.888826 w_k1_INTC_s013
                                 + 0.734821 w_k1_KO_s013
                                 + 0.881197 w_k1_MAT_s013
                                 + 0.81443 w_k1_MSFT_s013
                                 + 0.900778 w_k1_PG_s013 >= 1
 z_k1_s0140_constraint:          var_1 + z_k1_s0140 + 1.47123 w_k1_ADM_s014
                                 + 1.14708 w_k1_AXP_s014
                                 + 1.42252 w_k1_DIS_s014
                                 + 1.40575 w_k1_FDX_s014
                                 + 0.765636 w_k1_IBM_s014
                                 + 1.28355 w_k1_INTC_s014
                                 + 1.12586 w_k1_KO_s014
                                 + 0.751368 w_k1_MAT_s014
                                 + 1.28583 w_k1_MSFT_s014
                                 + 1.11427 w_k1_PG_s014 >= 1
 z_k1_s0141_constraint:          var_1 + z_k1_s0141 + 1.27331 w_k1_ADM_s014
                                 + 1.24264 w_k1_AXP_s014
                                 + 1.51273 w_k1_DIS_s014
                                 + 1.14933 w_k1_FDX_s014
                                 + 1.56221 w_k1_IBM_s014
                                 + 0.977766 w_k1_INTC_s014
                                 + 1.26156 w_k1_KO_s014 + 1.63724 w_k1_MAT_s014
                                 + 1.20616 w_k1_MSFT_s014
                                 + 1.17372 w_k1_PG_s014 >= 1
 z_k1_s0142_constraint:          var_1 + z_k1_s0142 + 1.23336 w_k1_ADM_s014
                                 + 0.904519 w_k1_AXP_s014
                                 + 1.02751 w_k1_DIS_s014
                                 + 1.00559 w_k1_FDX_s014
                                 + 0.919254 w_k1_IBM_s014
                                 + 0.790324 w_k1_INTC_s014
                                 + 1.0268 w_k1_KO_s014 + 0.794644 w_k1_MAT_s014
                                 + 1.0203 w_k1_MSFT_s014
                                 + 0.957166 w_k1_PG_s014 >= 1
 z_k1_s0143_constraint:          var_1 + z_k1_s0143 + 0.780724 w_k1_ADM_s014
                                 + 0.895957 w_k1_AXP_s014
                                 + 0.817118 w_k1_DIS_s014
                                 + 0.858557 w_k1_FDX_s014
                                 + 1.1358 w_k1_IBM_s014
                                 + 0.88188 w_k1_INTC_s014
                                 + 0.939433 w_k1_KO_s014
                                 + 1.12303 w_k1_MAT_s014
                                 + 0.863224 w_k1_MSFT_s014
                                 + 0.895903 w_k1_PG_s014 >= 1
 z_k1_s0144_constraint:          var_1 + z_k1_s0144 + 1.06198 w_k1_ADM_s014
                                 + 0.764235 w_k1_AXP_s014
                                 + 1.3216 w_k1_DIS_s014 + 1.13155 w_k1_FDX_s014
                                 + 0.87754 w_k1_IBM_s014
                                 + 1.0682 w_k1_INTC_s014
                                 + 0.986943 w_k1_KO_s014
                                 + 0.633892 w_k1_MAT_s014
                                 + 1.22324 w_k1_MSFT_s014
                                 + 1.02108 w_k1_PG_s014 >= 1
 z_k1_s0145_constraint:          var_1 + z_k1_s0145 + 0.969419 w_k1_ADM_s014
                                 + 0.960917 w_k1_AXP_s014
                                 + 0.782577 w_k1_DIS_s014
                                 + 0.978129 w_k1_FDX_s014
                                 + 0.78292 w_k1_IBM_s014
                                 + 0.864698 w_k1_INTC_s014
                                 + 0.822593 w_k1_KO_s014
                                 + 0.818457 w_k1_MAT_s014
                                 + 1.01925 w_k1_MSFT_s014
                                 + 0.919833 w_k1_PG_s014 >= 1
 z_k1_s0146_constraint:          var_1 + z_k1_s0146 + 1.29419 w_k1_ADM_s014
                                 + 1.11547 w_k1_AXP_s014
                                 + 1.21344 w_k1_DIS_s014
                                 + 1.11436 w_k1_FDX_s014
                                 + 1.14562 w_k1_IBM_s014
                                 + 0.958123 w_k1_INTC_s014
                                 + 1.15612 w_k1_KO_s014 + 1.04416 w_k1_MAT_s014
                                 + 1.04321 w_k1_MSFT_s014
                                 + 1.23318 w_k1_PG_s014 >= 1
 z_k1_s0147_constraint:          var_1 + z_k1_s0147 + 1.63599 w_k1_ADM_s014
                                 + 1.45915 w_k1_AXP_s014
                                 + 1.18776 w_k1_DIS_s014
                                 + 1.34294 w_k1_FDX_s014
                                 + 1.16645 w_k1_IBM_s014
                                 + 1.09831 w_k1_INTC_s014
                                 + 1.11342 w_k1_KO_s014
                                 + 0.808179 w_k1_MAT_s014
                                 + 1.11142 w_k1_MSFT_s014
                                 + 1.03046 w_k1_PG_s014 >= 1
 z_k1_s0150_constraint:          var_1 + z_k1_s0150 + 1.1976 w_k1_ADM_s015
                                 + 1.19004 w_k1_AXP_s015
                                 + 1.27385 w_k1_DIS_s015
                                 + 1.16026 w_k1_FDX_s015
                                 + 0.918666 w_k1_IBM_s015
                                 + 1.07717 w_k1_INTC_s015
                                 + 1.08474 w_k1_KO_s015
                                 + 0.845656 w_k1_MAT_s015
                                 + 1.20653 w_k1_MSFT_s015
                                 + 1.02419 w_k1_PG_s015 >= 1
 z_k1_s0151_constraint:          var_1 + z_k1_s0151 + 1.36348 w_k1_ADM_s015
                                 + 1.46458 w_k1_AXP_s015
                                 + 1.43113 w_k1_DIS_s015
                                 + 1.34235 w_k1_FDX_s015
                                 + 0.756629 w_k1_IBM_s015
                                 + 1.19473 w_k1_INTC_s015
                                 + 0.922587 w_k1_KO_s015
                                 + 0.791082 w_k1_MAT_s015
                                 + 1.14653 w_k1_MSFT_s015
                                 + 1.14642 w_k1_PG_s015 >= 1
 z_k1_s0152_constraint:          var_1 + z_k1_s0152 + 1.06694 w_k1_ADM_s015
                                 + 0.876625 w_k1_AXP_s015
                                 + 0.757514 w_k1_DIS_s015
                                 + 0.851433 w_k1_FDX_s015
                                 + 0.721564 w_k1_IBM_s015
                                 + 0.837631 w_k1_INTC_s015
                                 + 0.747105 w_k1_KO_s015
                                 + 0.508702 w_k1_MAT_s015
                                 + 0.845203 w_k1_MSFT_s015
                                 + 0.786706 w_k1_PG_s015 >= 1
 z_k1_s0153_constraint:          var_1 + z_k1_s0153 + 1.11253 w_k1_ADM_s015
                                 + 1.41524 w_k1_AXP_s015
                                 + 0.880167 w_k1_DIS_s015
                                 + 0.804161 w_k1_FDX_s015
                                 + 1.86531 w_k1_IBM_s015
                                 + 0.666357 w_k1_INTC_s015
                                 + 1.16158 w_k1_KO_s015 + 1.5674 w_k1_MAT_s015
                                 + 0.751309 w_k1_MSFT_s015
                                 + 0.950483 w_k1_PG_s015 >= 1
 z_k1_s0154_constraint:          var_1 + z_k1_s0154 + 1.18092 w_k1_ADM_s015
                                 + 0.849592 w_k1_AXP_s015
                                 + 0.70458 w_k1_DIS_s015
                                 + 0.776108 w_k1_FDX_s015
                                 + 1.32763 w_k1_IBM_s015
                                 + 0.68227 w_k1_INTC_s015
                                 + 1.01059 w_k1_KO_s015 + 1.02818 w_k1_MAT_s015
                                 + 0.746694 w_k1_MSFT_s015
                                 + 0.895624 w_k1_PG_s015 >= 1
 z_k1_s0155_constraint:          var_1 + z_k1_s0155 + 1.11584 w_k1_ADM_s015
                                 + 1.16403 w_k1_AXP_s015
                                 + 0.918515 w_k1_DIS_s015
                                 + 0.908127 w_k1_FDX_s015
                                 + 1.06185 w_k1_IBM_s015
                                 + 0.796358 w_k1_INTC_s015
                                 + 1.02725 w_k1_KO_s015 + 1.21673 w_k1_MAT_s015
                                 + 0.784804 w_k1_MSFT_s015
                                 + 1.00177 w_k1_PG_s015 >= 1
 z_k1_s0156_constraint:          var_1 + z_k1_s0156 + 1.03484 w_k1_ADM_s015
                                 + 1.51402 w_k1_AXP_s015
                                 + 1.14669 w_k1_DIS_s015
                                 + 1.14133 w_k1_FDX_s015
                                 + 1.44294 w_k1_IBM_s015
                                 + 1.16704 w_k1_INTC_s015
                                 + 1.16318 w_k1_KO_s015 + 1.37594 w_k1_MAT_s015
                                 + 0.949215 w_k1_MSFT_s015
                                 + 1.15858 w_k1_PG_s015 >= 1
 z_k1_s0157_constraint:          var_1 + z_k1_s0157 + 1.79329 w_k1_ADM_s015
                                 + 1.80181 w_k1_AXP_s015
                                 + 1.68059 w_k1_DIS_s015
                                 + 1.39251 w_k1_FDX_s015 + 1.8113 w_k1_IBM_s015
                                 + 1.02487 w_k1_INTC_s015
                                 + 1.59228 w_k1_KO_s015 + 2.14294 w_k1_MAT_s015
                                 + 1.06935 w_k1_MSFT_s015
                                 + 1.34329 w_k1_PG_s015 >= 1
 z_k1_s0160_constraint:          var_1 + z_k1_s0160 + 1.08983 w_k1_ADM_s016
                                 + 1.11612 w_k1_AXP_s016
                                 + 0.958321 w_k1_DIS_s016
                                 + 0.978424 w_k1_FDX_s016
                                 + 1.25006 w_k1_IBM_s016
                                 + 0.996418 w_k1_INTC_s016
                                 + 1.15716 w_k1_KO_s016 + 1.23301 w_k1_MAT_s016
                                 + 1.0176 w_k1_MSFT_s016 + 1.12626 w_k1_PG_s016
                                 >= 1
 z_k1_s0161_constraint:          var_1 + z_k1_s0161 + 1.03557 w_k1_ADM_s016
                                 + 1.43534 w_k1_AXP_s016
                                 + 1.06787 w_k1_DIS_s016
                                 + 1.12081 w_k1_FDX_s016
                                 + 1.03843 w_k1_IBM_s016
                                 + 0.958898 w_k1_INTC_s016
                                 + 0.874194 w_k1_KO_s016
                                 + 1.17109 w_k1_MAT_s016
                                 + 0.880229 w_k1_MSFT_s016
                                 + 1.09865 w_k1_PG_s016 >= 1
 z_k1_s0162_constraint:          var_1 + z_k1_s0162 + 1.17165 w_k1_ADM_s016
                                 + 1.11397 w_k1_AXP_s016
                                 + 1.21539 w_k1_DIS_s016
                                 + 1.18318 w_k1_FDX_s016
                                 + 1.11729 w_k1_IBM_s016
                                 + 1.09746 w_k1_INTC_s016
                                 + 1.09368 w_k1_KO_s016 + 1.05893 w_k1_MAT_s016
                                 + 1.23152 w_k1_MSFT_s016
                                 + 1.16485 w_k1_PG_s016 >= 1
 z_k1_s0163_constraint:          var_1 + z_k1_s0163 + 1.18835 w_k1_ADM_s016
                                 + 1.24148 w_k1_AXP_s016
                                 + 1.04023 w_k1_DIS_s016
                                 + 1.19186 w_k1_FDX_s016
                                 + 0.924828 w_k1_IBM_s016
                                 + 1.26605 w_k1_INTC_s016
                                 + 1.02472 w_k1_KO_s016 + 1.22768 w_k1_MAT_s016
                                 + 1.26642 w_k1_MSFT_s016
                                 + 1.10737 w_k1_PG_s016 >= 1
 z_k1_s0164_constraint:          var_1 + z_k1_s0164 + 1.00323 w_k1_ADM_s016
                                 + 0.896773 w_k1_AXP_s016
                                 + 1.06418 w_k1_DIS_s016
                                 + 0.883168 w_k1_FDX_s016
                                 + 1.32774 w_k1_IBM_s016
                                 + 0.862121 w_k1_INTC_s016
                                 + 0.977245 w_k1_KO_s016 + 1.1566 w_k1_MAT_s016
                                 + 1.04561 w_k1_MSFT_s016
                                 + 0.983489 w_k1_PG_s016 >= 1
 z_k1_s0165_constraint:          var_1 + z_k1_s0165 + 0.77502 w_k1_ADM_s016
                                 + 0.801176 w_k1_AXP_s016
                                 + 0.646432 w_k1_DIS_s016
                                 + 0.775748 w_k1_FDX_s016
                                 + 0.985345 w_k1_IBM_s016
                                 + 0.835304 w_k1_INTC_s016
                                 + 0.822769 w_k1_KO_s016
                                 + 0.780379 w_k1_MAT_s016
                                 + 0.803001 w_k1_MSFT_s016
                                 + 0.866159 w_k1_PG_s016 >= 1
 z_k1_s0166_constraint:          var_1 + z_k1_s0166 + 0.777373 w_k1_ADM_s016
                                 + 0.876016 w_k1_AXP_s016
                                 + 0.801157 w_k1_DIS_s016
                                 + 0.865748 w_k1_FDX_s016
                                 + 0.774475 w_k1_IBM_s016
                                 + 0.907803 w_k1_INTC_s016
                                 + 0.755302 w_k1_KO_s016
                                 + 0.816014 w_k1_MAT_s016
                                 + 0.878049 w_k1_MSFT_s016
                                 + 0.916899 w_k1_PG_s016 >= 1
 z_k1_s0167_constraint:          var_1 + z_k1_s0167 + 1.20162 w_k1_ADM_s016
                                 + 1.10387 w_k1_AXP_s016
                                 + 1.56183 w_k1_DIS_s016
                                 + 1.35889 w_k1_FDX_s016
                                 + 0.955379 w_k1_IBM_s016
                                 + 1.14603 w_k1_INTC_s016
                                 + 1.10795 w_k1_KO_s016 + 1.12465 w_k1_MAT_s016
                                 + 1.519 w_k1_MSFT_s016 + 1.22675 w_k1_PG_s016
                                 >= 1
 z_k1_s0170_constraint:          var_1 + z_k1_s0170 + 0.756721 w_k1_ADM_s017
                                 + 0.825691 w_k1_AXP_s017
                                 + 0.782408 w_k1_DIS_s017
                                 + 0.837829 w_k1_FDX_s017
                                 + 1.19626 w_k1_IBM_s017
                                 + 0.978592 w_k1_INTC_s017
                                 + 0.969895 w_k1_KO_s017
                                 + 1.26324 w_k1_MAT_s017
                                 + 0.898035 w_k1_MSFT_s017
                                 + 0.869141 w_k1_PG_s017 >= 1
 z_k1_s0171_constraint:          var_1 + z_k1_s0171 + 0.876981 w_k1_ADM_s017
                                 + 0.68179 w_k1_AXP_s017
                                 + 0.595304 w_k1_DIS_s017
                                 + 0.886619 w_k1_FDX_s017
                                 + 0.609347 w_k1_IBM_s017
                                 + 0.92388 w_k1_INTC_s017
                                 + 0.682911 w_k1_KO_s017
                                 + 0.583718 w_k1_MAT_s017
                                 + 0.869414 w_k1_MSFT_s017
                                 + 0.86807 w_k1_PG_s017 >= 1
 z_k1_s0172_constraint:          var_1 + z_k1_s0172 + 0.713689 w_k1_ADM_s017
                                 + 0.75455 w_k1_AXP_s017
                                 + 0.728163 w_k1_DIS_s017
                                 + 0.910604 w_k1_FDX_s017
                                 + 0.823341 w_k1_IBM_s017
                                 + 0.925287 w_k1_INTC_s017
                                 + 0.789216 w_k1_KO_s017
                                 + 0.798619 w_k1_MAT_s017
                                 + 0.834791 w_k1_MSFT_s017
                                 + 0.930446 w_k1_PG_s017 >= 1
 z_k1_s0173_constraint:          var_1 + z_k1_s0173 + 1.17374 w_k1_ADM_s017
                                 + 1.27867 w_k1_AXP_s017
                                 + 1.50351 w_k1_DIS_s017
                                 + 1.34266 w_k1_FDX_s017
                                 + 0.995152 w_k1_IBM_s017
                                 + 1.28907 w_k1_INTC_s017
                                 + 1.18489 w_k1_KO_s017 + 1.12719 w_k1_MAT_s017
                                 + 1.20943 w_k1_MSFT_s017 + 1.0213 w_k1_PG_s017
                                 >= 1
 z_k1_s0174_constraint:          var_1 + z_k1_s0174 + 1.04519 w_k1_ADM_s017
                                 + 0.982607 w_k1_AXP_s017
                                 + 1.3095 w_k1_DIS_s017 + 1.19971 w_k1_FDX_s017
                                 + 1.23198 w_k1_IBM_s017
                                 + 1.13662 w_k1_INTC_s017
                                 + 1.14934 w_k1_KO_s017 + 1.04736 w_k1_MAT_s017
                                 + 1.09391 w_k1_MSFT_s017
                                 + 1.02602 w_k1_PG_s017 >= 1
 z_k1_s0175_constraint:          var_1 + z_k1_s0175 + 0.991244 w_k1_ADM_s017
                                 + 0.974322 w_k1_AXP_s017
                                 + 1.05075 w_k1_DIS_s017 + 1.1768 w_k1_FDX_s017
                                 + 0.740703 w_k1_IBM_s017
                                 + 1.32805 w_k1_INTC_s017
                                 + 0.823687 w_k1_KO_s017
                                 + 0.865006 w_k1_MAT_s017
                                 + 0.952479 w_k1_MSFT_s017
                                 + 0.979563 w_k1_PG_s017 >= 1
 z_k1_s0176_constraint:          var_1 + z_k1_s0176 + 0.921983 w_k1_ADM_s017
                                 + 1.22961 w_k1_AXP_s017
                                 + 0.968827 w_k1_DIS_s017
                                 + 1.20377 w_k1_FDX_s017
                                 + 0.976945 w_k1_IBM_s017
                                 + 1.05463 w_k1_INTC_s017
                                 + 0.879824 w_k1_KO_s017
                                 + 1.19403 w_k1_MAT_s017
                                 + 1.01897 w_k1_MSFT_s017
                                 + 1.06624 w_k1_PG_s017 >= 1
 z_k1_s0177_constraint:          var_1 + z_k1_s0177 + 0.867832 w_k1_ADM_s017
                                 + 1.28847 w_k1_AXP_s017
                                 + 1.33477 w_k1_DIS_s017
                                 + 1.12926 w_k1_FDX_s017
                                 + 1.72379 w_k1_IBM_s017
                                 + 1.22481 w_k1_INTC_s017
                                 + 1.29325 w_k1_KO_s017 + 1.31383 w_k1_MAT_s017
                                 + 1.17623 w_k1_MSFT_s017
                                 + 1.06584 w_k1_PG_s017 >= 1
 z_k1_s1000_constraint:          var_1 + z_k1_s1000 + 1.42286 w_k1_ADM_s100
                                 + 1.51832 w_k1_AXP_s100 + 1.6598 w_k1_DIS_s100
                                 + 1.48008 w_k1_FDX_s100
                                 + 1.30349 w_k1_IBM_s100
                                 + 1.17195 w_k1_INTC_s100 + 1.2404 w_k1_KO_s100
                                 + 1.34402 w_k1_MAT_s100
                                 + 1.22138 w_k1_MSFT_s100 + 1.2827 w_k1_PG_s100
                                 >= 1
 z_k1_s1001_constraint:          var_1 + z_k1_s1001 + 0.895161 w_k1_ADM_s100
                                 + 0.842549 w_k1_AXP_s100
                                 + 0.706344 w_k1_DIS_s100
                                 + 0.757799 w_k1_FDX_s100
                                 + 1.28823 w_k1_IBM_s100
                                 + 0.597094 w_k1_INTC_s100
                                 + 0.77995 w_k1_KO_s100 + 1.29316 w_k1_MAT_s100
                                 + 0.586541 w_k1_MSFT_s100
                                 + 0.879819 w_k1_PG_s100 >= 1
 z_k1_s1002_constraint:          var_1 + z_k1_s1002 + 0.90148 w_k1_ADM_s100
                                 + 0.953198 w_k1_AXP_s100
                                 + 1.10641 w_k1_DIS_s100
                                 + 1.03809 w_k1_FDX_s100
                                 + 1.11275 w_k1_IBM_s100
                                 + 1.00505 w_k1_INTC_s100
                                 + 1.07612 w_k1_KO_s100 + 1.12572 w_k1_MAT_s100
                                 + 0.974458 w_k1_MSFT_s100
                                 + 1.05106 w_k1_PG_s100 >= 1
 z_k1_s1003_constraint:          var_1 + z_k1_s1003 + 0.898802 w_k1_ADM_s100
                                 + 1.2026 w_k1_AXP_s100 + 1.1558 w_k1_DIS_s100
                                 + 0.954841 w_k1_FDX_s100
                                 + 1.70675 w_k1_IBM_s100
                                 + 0.724006 w_k1_INTC_s100
                                 + 1.16745 w_k1_KO_s100 + 1.88705 w_k1_MAT_s100
                                 + 0.88176 w_k1_MSFT_s100 + 1.1032 w_k1_PG_s100
                                 >= 1
 z_k1_s1004_constraint:          var_1 + z_k1_s1004 + 0.75387 w_k1_ADM_s100
                                 + 0.764942 w_k1_AXP_s100
                                 + 0.758064 w_k1_DIS_s100
                                 + 0.736883 w_k1_FDX_s100
                                 + 0.949553 w_k1_IBM_s100
                                 + 0.749761 w_k1_INTC_s100
                                 + 0.786262 w_k1_KO_s100 + 1.0819 w_k1_MAT_s100
                                 + 0.757371 w_k1_MSFT_s100
                                 + 0.829145 w_k1_PG_s100 >= 1
 z_k1_s1005_constraint:          var_1 + z_k1_s1005 + 0.868405 w_k1_ADM_s100
                                 + 0.881189 w_k1_AXP_s100
                                 + 0.798367 w_k1_DIS_s100
                                 + 0.94458 w_k1_FDX_s100
                                 + 0.746588 w_k1_IBM_s100
                                 + 0.872553 w_k1_INTC_s100
                                 + 0.876222 w_k1_KO_s100
                                 + 0.742004 w_k1_MAT_s100
                                 + 0.829254 w_k1_MSFT_s100
                                 + 0.865927 w_k1_PG_s100 >= 1
 z_k1_s1006_constraint:          var_1 + z_k1_s1006 + 1.34073 w_k1_ADM_s100
                                 + 1.18009 w_k1_AXP_s100
                                 + 1.45977 w_k1_DIS_s100
                                 + 1.31086 w_k1_FDX_s100
                                 + 0.817908 w_k1_IBM_s100
                                 + 0.800486 w_k1_INTC_s100
                                 + 0.985859 w_k1_KO_s100
                                 + 1.01896 w_k1_MAT_s100
                                 + 1.12747 w_k1_MSFT_s100 + 1.1489 w_k1_PG_s100
                                 >= 1
 z_k1_s1007_constraint:          var_1 + z_k1_s1007 + 1.36278 w_k1_ADM_s100
                                 + 1.2313 w_k1_AXP_s100
                                 + 0.975113 w_k1_DIS_s100
                                 + 1.05756 w_k1_FDX_s100
                                 + 1.09315 w_k1_IBM_s100
                                 + 0.783021 w_k1_INTC_s100
                                 + 0.858183 w_k1_KO_s100
                                 + 1.34856 w_k1_MAT_s100
                                 + 0.946591 w_k1_MSFT_s100
                                 + 1.03611 w_k1_PG_s100 >= 1
 z_k1_s1010_constraint:          var_1 + z_k1_s1010 + 0.808083 w_k1_ADM_s101
                                 + 0.742596 w_k1_AXP_s101
                                 + 0.836977 w_k1_DIS_s101
                                 + 0.783577 w_k1_FDX_s101
                                 + 1.17417 w_k1_IBM_s101
                                 + 0.83201 w_k1_INTC_s101
                                 + 0.895975 w_k1_KO_s101
                                 + 0.970379 w_k1_MAT_s101
                                 + 0.917989 w_k1_MSFT_s101
                                 + 0.874626 w_k1_PG_s101 >= 1
 z_k1_s1011_constraint:          var_1 + z_k1_s1011 + 0.919905 w_k1_ADM_s101
                                 + 1.18952 w_k1_AXP_s101
                                 + 1.04112 w_k1_DIS_s101
                                 + 1.08182 w_k1_FDX_s101
                                 + 1.35556 w_k1_IBM_s101
                                 + 0.861543 w_k1_INTC_s101
                                 + 1.04287 w_k1_KO_s101 + 1.55463 w_k1_MAT_s101
                                 + 1.00591 w_k1_MSFT_s101
                                 + 1.12137 w_k1_PG_s101 >= 1
 z_k1_s1012_constraint:          var_1 + z_k1_s1012 + 0.78564 w_k1_ADM_s101
                                 + 0.912671 w_k1_AXP_s101
                                 + 1.31153 w_k1_DIS_s101
                                 + 1.04846 w_k1_FDX_s101 + 1.2178 w_k1_IBM_s101
                                 + 1.02324 w_k1_INTC_s101
                                 + 1.12824 w_k1_KO_s101 + 1.23394 w_k1_MAT_s101
                                 + 1.06162 w_k1_MSFT_s101
                                 + 1.06624 w_k1_PG_s101 >= 1
 z_k1_s1013_constraint:          var_1 + z_k1_s1013 + 0.999276 w_k1_ADM_s101
                                 + 0.849986 w_k1_AXP_s101
                                 + 1.02171 w_k1_DIS_s101
                                 + 1.01887 w_k1_FDX_s101 + 0.745 w_k1_IBM_s101
                                 + 0.879918 w_k1_INTC_s101
                                 + 0.860825 w_k1_KO_s101
                                 + 0.533731 w_k1_MAT_s101
                                 + 0.907119 w_k1_MSFT_s101
                                 + 0.957318 w_k1_PG_s101 >= 1
 z_k1_s1014_constraint:          var_1 + z_k1_s1014 + 1.4974 w_k1_ADM_s101
                                 + 1.2724 w_k1_AXP_s101 + 1.51975 w_k1_DIS_s101
                                 + 1.29476 w_k1_FDX_s101
                                 + 1.15205 w_k1_IBM_s101
                                 + 1.0691 w_k1_INTC_s101 + 1.18655 w_k1_KO_s101
                                 + 0.978641 w_k1_MAT_s101
                                 + 1.20032 w_k1_MSFT_s101
                                 + 1.33338 w_k1_PG_s101 >= 1
 z_k1_s1015_constraint:          var_1 + z_k1_s1015 + 1.44998 w_k1_ADM_s101
                                 + 1.55486 w_k1_AXP_s101
                                 + 2.12224 w_k1_DIS_s101
                                 + 1.56277 w_k1_FDX_s101
                                 + 1.21594 w_k1_IBM_s101
                                 + 1.01106 w_k1_INTC_s101 + 1.2865 w_k1_KO_s101
                                 + 1.41911 w_k1_MAT_s101
                                 + 1.35993 w_k1_MSFT_s101
                                 + 1.25609 w_k1_PG_s101 >= 1
 z_k1_s1016_constraint:          var_1 + z_k1_s1016 + 0.65534 w_k1_ADM_s101
                                 + 0.549193 w_k1_AXP_s101
                                 + 0.679432 w_k1_DIS_s101
                                 + 0.807599 w_k1_FDX_s101
                                 + 0.763794 w_k1_IBM_s101
                                 + 0.688256 w_k1_INTC_s101
                                 + 0.790488 w_k1_KO_s101
                                 + 0.684087 w_k1_MAT_s101
                                 + 0.748484 w_k1_MSFT_s101
                                 + 0.83888 w_k1_PG_s101 >= 1
 z_k1_s1017_constraint:          var_1 + z_k1_s1017 + 0.861429 w_k1_ADM_s101
                                 + 1.21018 w_k1_AXP_s101
                                 + 0.794296 w_k1_DIS_s101
                                 + 0.894566 w_k1_FDX_s101
                                 + 1.3735 w_k1_IBM_s101
                                 + 0.757148 w_k1_INTC_s101
                                 + 0.887941 w_k1_KO_s101
                                 + 1.50111 w_k1_MAT_s101
                                 + 0.828552 w_k1_MSFT_s101
                                 + 0.948046 w_k1_PG_s101 >= 1
 z_k1_s1020_constraint:          var_1 + z_k1_s1020 + 0.846587 w_k1_ADM_s102
                                 + 0.975845 w_k1_AXP_s102
                                 + 0.920612 w_k1_DIS_s102
                                 + 0.906393 w_k1_FDX_s102
                                 + 1.03388 w_k1_IBM_s102
                                 + 0.897137 w_k1_INTC_s102
                                 + 0.86617 w_k1_KO_s102 + 1.06002 w_k1_MAT_s102
                                 + 0.845253 w_k1_MSFT_s102
                                 + 0.855829 w_k1_PG_s102 >= 1
 z_k1_s1021_constraint:          var_1 + z_k1_s1021 + 0.635236 w_k1_ADM_s102
                                 + 0.656626 w_k1_AXP_s102
                                 + 0.530068 w_k1_DIS_s102
                                 + 0.640425 w_k1_FDX_s102
                                 + 0.788252 w_k1_IBM_s102
                                 + 0.670815 w_k1_INTC_s102
                                 + 0.686106 w_k1_KO_s102
                                 + 0.707179 w_k1_MAT_s102
                                 + 0.708627 w_k1_MSFT_s102
                                 + 0.692409 w_k1_PG_s102 >= 1
 z_k1_s1022_constraint:          var_1 + z_k1_s1022 + 1.1703 w_k1_ADM_s102
                                 + 1.50388 w_k1_AXP_s102
                                 + 1.77254 w_k1_DIS_s102
                                 + 1.51588 w_k1_FDX_s102
                                 + 1.48508 w_k1_IBM_s102
                                 + 1.41378 w_k1_INTC_s102
                                 + 1.61409 w_k1_KO_s102 + 2.59797 w_k1_MAT_s102
                                 + 1.52959 w_k1_MSFT_s102
                                 + 1.34006 w_k1_PG_s102 >= 1
 z_k1_s1023_constraint:          var_1 + z_k1_s1023 + 1.43967 w_k1_ADM_s102
                                 + 1.43842 w_k1_AXP_s102
                                 + 1.66649 w_k1_DIS_s102
                                 + 1.53742 w_k1_FDX_s102
                                 + 1.04692 w_k1_IBM_s102
                                 + 1.21183 w_k1_INTC_s102
                                 + 1.26345 w_k1_KO_s102 + 1.05303 w_k1_MAT_s102
                                 + 1.5304 w_k1_MSFT_s102 + 1.16118 w_k1_PG_s102
                                 >= 1
 z_k1_s1024_constraint:          var_1 + z_k1_s1024 + 1.01008 w_k1_ADM_s102
                                 + 1.06291 w_k1_AXP_s102
                                 + 1.13668 w_k1_DIS_s102
                                 + 1.15435 w_k1_FDX_s102
                                 + 0.92779 w_k1_IBM_s102
                                 + 1.08226 w_k1_INTC_s102
                                 + 1.05562 w_k1_KO_s102 + 1.10395 w_k1_MAT_s102
                                 + 1.16143 w_k1_MSFT_s102 + 1.1052 w_k1_PG_s102
                                 >= 1
 z_k1_s1025_constraint:          var_1 + z_k1_s1025 + 0.725824 w_k1_ADM_s102
                                 + 0.67811 w_k1_AXP_s102
                                 + 0.773164 w_k1_DIS_s102
                                 + 0.77407 w_k1_FDX_s102
                                 + 1.32354 w_k1_IBM_s102
                                 + 0.885265 w_k1_INTC_s102
                                 + 1.05812 w_k1_KO_s102
                                 + 0.930695 w_k1_MAT_s102
                                 + 0.780329 w_k1_MSFT_s102
                                 + 0.887433 w_k1_PG_s102 >= 1
 z_k1_s1026_constraint:          var_1 + z_k1_s1026 + 1.06155 w_k1_ADM_s102
                                 + 1.41315 w_k1_AXP_s102
                                 + 1.20433 w_k1_DIS_s102
                                 + 1.03743 w_k1_FDX_s102
                                 + 1.58274 w_k1_IBM_s102
                                 + 1.03584 w_k1_INTC_s102
                                 + 1.23816 w_k1_KO_s102 + 1.35092 w_k1_MAT_s102
                                 + 0.94988 w_k1_MSFT_s102
                                 + 1.02437 w_k1_PG_s102 >= 1
 z_k1_s1027_constraint:          var_1 + z_k1_s1027 + 0.819179 w_k1_ADM_s102
                                 + 0.838239 w_k1_AXP_s102
                                 + 0.866532 w_k1_DIS_s102
                                 + 0.928138 w_k1_FDX_s102
                                 + 0.727488 w_k1_IBM_s102
                                 + 0.988372 w_k1_INTC_s102
                                 + 0.833697 w_k1_KO_s102
                                 + 0.732012 w_k1_MAT_s102
                                 + 1.02182 w_k1_MSFT_s102
                                 + 0.918353 w_k1_PG_s102 >= 1
 z_k1_s1030_constraint:          var_1 + z_k1_s1030 + 1.10671 w_k1_ADM_s103
                                 + 0.969056 w_k1_AXP_s103
                                 + 1.14425 w_k1_DIS_s103
                                 + 1.22392 w_k1_FDX_s103
                                 + 0.80364 w_k1_IBM_s103
                                 + 1.20253 w_k1_INTC_s103
                                 + 1.15286 w_k1_KO_s103
                                 + 0.897067 w_k1_MAT_s103
                                 + 1.34327 w_k1_MSFT_s103
                                 + 1.17763 w_k1_PG_s103 >= 1
 z_k1_s1031_constraint:          var_1 + z_k1_s1031 + 1.09211 w_k1_ADM_s103
                                 + 1.06997 w_k1_AXP_s103
                                 + 0.925106 w_k1_DIS_s103
                                 + 1.01225 w_k1_FDX_s103
                                 + 0.883196 w_k1_IBM_s103
                                 + 1.10349 w_k1_INTC_s103
                                 + 1.04462 w_k1_KO_s103 + 1.11591 w_k1_MAT_s103
                                 + 1.11754 w_k1_MSFT_s103
                                 + 1.01416 w_k1_PG_s103 >= 1
 z_k1_s1032_constraint:          var_1 + z_k1_s1032 + 0.785573 w_k1_ADM_s103
                                 + 0.793531 w_k1_AXP_s103
                                 + 0.960115 w_k1_DIS_s103
                                 + 0.889271 w_k1_FDX_s103
                                 + 0.958395 w_k1_IBM_s103
                                 + 0.948997 w_k1_INTC_s103
                                 + 1.03081 w_k1_KO_s103
                                 + 0.967875 w_k1_MAT_s103
                                 + 1.10332 w_k1_MSFT_s103
                                 + 0.914611 w_k1_PG_s103 >= 1
 z_k1_s1033_constraint:          var_1 + z_k1_s1033 + 0.709073 w_k1_ADM_s103
                                 + 0.668939 w_k1_AXP_s103
                                 + 0.672524 w_k1_DIS_s103
                                 + 0.80736 w_k1_FDX_s103
                                 + 0.985113 w_k1_IBM_s103
                                 + 0.892887 w_k1_INTC_s103
                                 + 0.845866 w_k1_KO_s103
                                 + 1.39693 w_k1_MAT_s103
                                 + 0.865499 w_k1_MSFT_s103
                                 + 0.893881 w_k1_PG_s103 >= 1
 z_k1_s1034_constraint:          var_1 + z_k1_s1034 + 0.984016 w_k1_ADM_s103
                                 + 1.15457 w_k1_AXP_s103 + 1.0425 w_k1_DIS_s103
                                 + 1.04733 w_k1_FDX_s103
                                 + 1.30798 w_k1_IBM_s103
                                 + 1.24284 w_k1_INTC_s103
                                 + 1.22555 w_k1_KO_s103 + 1.76617 w_k1_MAT_s103
                                 + 1.23725 w_k1_MSFT_s103
                                 + 1.03146 w_k1_PG_s103 >= 1
 z_k1_s1035_constraint:          var_1 + z_k1_s1035 + 1.05201 w_k1_ADM_s103
                                 + 1.5538 w_k1_AXP_s103 + 1.31747 w_k1_DIS_s103
                                 + 1.29652 w_k1_FDX_s103
                                 + 1.31913 w_k1_IBM_s103
                                 + 1.21616 w_k1_INTC_s103
                                 + 1.33593 w_k1_KO_s103 + 1.90676 w_k1_MAT_s103
                                 + 1.25728 w_k1_MSFT_s103
                                 + 1.21248 w_k1_PG_s103 >= 1
 z_k1_s1036_constraint:          var_1 + z_k1_s1036 + 1.24339 w_k1_ADM_s103
                                 + 1.6511 w_k1_AXP_s103 + 1.80318 w_k1_DIS_s103
                                 + 1.50932 w_k1_FDX_s103
                                 + 1.49591 w_k1_IBM_s103
                                 + 1.24639 w_k1_INTC_s103
                                 + 1.55766 w_k1_KO_s103 + 1.8615 w_k1_MAT_s103
                                 + 1.52771 w_k1_MSFT_s103
                                 + 1.43885 w_k1_PG_s103 >= 1
 z_k1_s1037_constraint:          var_1 + z_k1_s1037 + 0.822695 w_k1_ADM_s103
                                 + 1.13332 w_k1_AXP_s103
                                 + 0.815689 w_k1_DIS_s103
                                 + 0.891025 w_k1_FDX_s103
                                 + 1.42357 w_k1_IBM_s103
                                 + 0.884636 w_k1_INTC_s103
                                 + 0.952307 w_k1_KO_s103
                                 + 1.93115 w_k1_MAT_s103
                                 + 0.903097 w_k1_MSFT_s103
                                 + 1.1389 w_k1_PG_s103 >= 1
 z_k1_s1040_constraint:          var_1 + z_k1_s1040 + 1.17028 w_k1_ADM_s104
                                 + 1.11795 w_k1_AXP_s104
                                 + 1.13409 w_k1_DIS_s104
                                 + 1.02863 w_k1_FDX_s104
                                 + 1.26397 w_k1_IBM_s104
                                 + 1.0716 w_k1_INTC_s104 + 1.15714 w_k1_KO_s104
                                 + 1.06128 w_k1_MAT_s104
                                 + 0.968291 w_k1_MSFT_s104
                                 + 1.19163 w_k1_PG_s104 >= 1
 z_k1_s1041_constraint:          var_1 + z_k1_s1041 + 0.896087 w_k1_ADM_s104
                                 + 0.84131 w_k1_AXP_s104
                                 + 0.823061 w_k1_DIS_s104
                                 + 0.915363 w_k1_FDX_s104
                                 + 1.01024 w_k1_IBM_s104
                                 + 0.938402 w_k1_INTC_s104
                                 + 0.939642 w_k1_KO_s104
                                 + 0.764596 w_k1_MAT_s104
                                 + 0.911923 w_k1_MSFT_s104
                                 + 0.956285 w_k1_PG_s104 >= 1
 z_k1_s1042_constraint:          var_1 + z_k1_s1042 + 1.0783 w_k1_ADM_s104
                                 + 0.851221 w_k1_AXP_s104
                                 + 0.82303 w_k1_DIS_s104
                                 + 0.936829 w_k1_FDX_s104
                                 + 0.74273 w_k1_IBM_s104
                                 + 1.06256 w_k1_INTC_s104
                                 + 0.862933 w_k1_KO_s104
                                 + 0.594027 w_k1_MAT_s104
                                 + 0.817956 w_k1_MSFT_s104
                                 + 0.88903 w_k1_PG_s104 >= 1
 z_k1_s1043_constraint:          var_1 + z_k1_s1043 + 0.803996 w_k1_ADM_s104
                                 + 0.926055 w_k1_AXP_s104
                                 + 0.833819 w_k1_DIS_s104
                                 + 0.817586 w_k1_FDX_s104
                                 + 1.32979 w_k1_IBM_s104
                                 + 0.933978 w_k1_INTC_s104
                                 + 0.9068 w_k1_KO_s104 + 1.05785 w_k1_MAT_s104
                                 + 0.823217 w_k1_MSFT_s104
                                 + 1.09219 w_k1_PG_s104 >= 1
 z_k1_s1044_constraint:          var_1 + z_k1_s1044 + 1.11863 w_k1_ADM_s104
                                 + 1.05648 w_k1_AXP_s104
                                 + 1.04691 w_k1_DIS_s104
                                 + 1.09926 w_k1_FDX_s104
                                 + 0.98577 w_k1_IBM_s104
                                 + 1.2407 w_k1_INTC_s104 + 1.05615 w_k1_KO_s104
                                 + 1.21955 w_k1_MAT_s104
                                 + 1.07594 w_k1_MSFT_s104
                                 + 1.11179 w_k1_PG_s104 >= 1
 z_k1_s1045_constraint:          var_1 + z_k1_s1045 + 1.14002 w_k1_ADM_s104
                                 + 1.28397 w_k1_AXP_s104
                                 + 1.30317 w_k1_DIS_s104
                                 + 1.23004 w_k1_FDX_s104
                                 + 1.31255 w_k1_IBM_s104
                                 + 1.3604 w_k1_INTC_s104 + 1.19032 w_k1_KO_s104
                                 + 1.33983 w_k1_MAT_s104
                                 + 1.21259 w_k1_MSFT_s104
                                 + 1.29931 w_k1_PG_s104 >= 1
 z_k1_s1046_constraint:          var_1 + z_k1_s1046 + 1.18469 w_k1_ADM_s104
                                 + 1.308 w_k1_AXP_s104 + 1.31718 w_k1_DIS_s104
                                 + 1.09315 w_k1_FDX_s104
                                 + 0.907354 w_k1_IBM_s104
                                 + 1.03561 w_k1_INTC_s104
                                 + 1.17109 w_k1_KO_s104 + 1.09124 w_k1_MAT_s104
                                 + 0.969332 w_k1_MSFT_s104
                                 + 1.17725 w_k1_PG_s104 >= 1
 z_k1_s1047_constraint:          var_1 + z_k1_s1047 + 1.49142 w_k1_ADM_s104
                                 + 1.42047 w_k1_AXP_s104
                                 + 1.78482 w_k1_DIS_s104
                                 + 1.35956 w_k1_FDX_s104
                                 + 1.37399 w_k1_IBM_s104
                                 + 1.28998 w_k1_INTC_s104
                                 + 1.38879 w_k1_KO_s104 + 1.42614 w_k1_MAT_s104
                                 + 1.23429 w_k1_MSFT_s104
                                 + 1.36369 w_k1_PG_s104 >= 1
 z_k1_s1050_constraint:          var_1 + z_k1_s1050 + 0.872393 w_k1_ADM_s105
                                 + 1.00459 w_k1_AXP_s105
                                 + 0.69873 w_k1_DIS_s105
                                 + 0.851245 w_k1_FDX_s105
                                 + 1.03576 w_k1_IBM_s105
                                 + 0.917502 w_k1_INTC_s105
                                 + 0.974973 w_k1_KO_s105
                                 + 1.04938 w_k1_MAT_s105
                                 + 0.777712 w_k1_MSFT_s105
                                 + 0.940083 w_k1_PG_s105 >= 1
 z_k1_s1051_constraint:          var_1 + z_k1_s1051 + 1.02063 w_k1_ADM_s105
                                 + 0.764291 w_k1_AXP_s105
                                 + 0.940623 w_k1_DIS_s105
                                 + 0.896507 w_k1_FDX_s105
                                 + 0.733743 w_k1_IBM_s105
                                 + 0.949364 w_k1_INTC_s105
                                 + 0.861008 w_k1_KO_s105
                                 + 0.491593 w_k1_MAT_s105
                                 + 0.839665 w_k1_MSFT_s105
                                 + 0.910787 w_k1_PG_s105 >= 1
 z_k1_s1052_constraint:          var_1 + z_k1_s1052 + 1.13882 w_k1_ADM_s105
                                 + 1.23807 w_k1_AXP_s105
                                 + 1.18614 w_k1_DIS_s105
                                 + 1.06135 w_k1_FDX_s105
                                 + 1.08072 w_k1_IBM_s105
                                 + 0.989654 w_k1_INTC_s105
                                 + 1.00612 w_k1_KO_s105 + 0.8737 w_k1_MAT_s105
                                 + 0.923026 w_k1_MSFT_s105
                                 + 1.07571 w_k1_PG_s105 >= 1
 z_k1_s1053_constraint:          var_1 + z_k1_s1053 + 1.16376 w_k1_ADM_s105
                                 + 0.933031 w_k1_AXP_s105
                                 + 1.20097 w_k1_DIS_s105
                                 + 1.21202 w_k1_FDX_s105
                                 + 0.78971 w_k1_IBM_s105
                                 + 1.20642 w_k1_INTC_s105
                                 + 1.12447 w_k1_KO_s105
                                 + 0.704921 w_k1_MAT_s105
                                 + 1.29203 w_k1_MSFT_s105
                                 + 1.03258 w_k1_PG_s105 >= 1
 z_k1_s1054_constraint:          var_1 + z_k1_s1054 + 0.67653 w_k1_ADM_s105
                                 + 0.658456 w_k1_AXP_s105
                                 + 0.622366 w_k1_DIS_s105
                                 + 0.719783 w_k1_FDX_s105
                                 + 0.761017 w_k1_IBM_s105
                                 + 0.693037 w_k1_INTC_s105
                                 + 0.715881 w_k1_KO_s105
                                 + 0.534435 w_k1_MAT_s105
                                 + 0.622696 w_k1_MSFT_s105
                                 + 0.797614 w_k1_PG_s105 >= 1
 z_k1_s1055_constraint:          var_1 + z_k1_s1055 + 1.52256 w_k1_ADM_s105
                                 + 1.66619 w_k1_AXP_s105 + 1.1418 w_k1_DIS_s105
                                 + 1.35356 w_k1_FDX_s105
                                 + 1.01751 w_k1_IBM_s105
                                 + 1.06973 w_k1_INTC_s105
                                 + 1.10089 w_k1_KO_s105
                                 + 0.928879 w_k1_MAT_s105
                                 + 1.06814 w_k1_MSFT_s105
                                 + 1.07564 w_k1_PG_s105 >= 1
 z_k1_s1056_constraint:          var_1 + z_k1_s1056 + 0.724464 w_k1_ADM_s105
                                 + 1.03436 w_k1_AXP_s105
                                 + 1.06748 w_k1_DIS_s105
                                 + 0.961924 w_k1_FDX_s105
                                 + 1.4888 w_k1_IBM_s105
                                 + 1.06422 w_k1_INTC_s105 + 1.2666 w_k1_KO_s105
                                 + 1.41758 w_k1_MAT_s105
                                 + 1.00878 w_k1_MSFT_s105
                                 + 1.14268 w_k1_PG_s105 >= 1
 z_k1_s1057_constraint:          var_1 + z_k1_s1057 + 1.42039 w_k1_ADM_s105
                                 + 1.4583 w_k1_AXP_s105 + 1.84646 w_k1_DIS_s105
                                 + 1.49183 w_k1_FDX_s105
                                 + 1.31982 w_k1_IBM_s105
                                 + 1.43058 w_k1_INTC_s105 + 1.4027 w_k1_KO_s105
                                 + 0.843849 w_k1_MAT_s105
                                 + 1.39597 w_k1_MSFT_s105
                                 + 1.15122 w_k1_PG_s105 >= 1
 z_k1_s1060_constraint:          var_1 + z_k1_s1060 + 0.833172 w_k1_ADM_s106
                                 + 0.573411 w_k1_AXP_s106
                                 + 0.604827 w_k1_DIS_s106
                                 + 0.659362 w_k1_FDX_s106
                                 + 0.753748 w_k1_IBM_s106
                                 + 0.685598 w_k1_INTC_s106
                                 + 0.620782 w_k1_KO_s106
                                 + 0.393179 w_k1_MAT_s106
                                 + 0.724925 w_k1_MSFT_s106
                                 + 0.759152 w_k1_PG_s106 >= 1
 z_k1_s1061_constraint:          var_1 + z_k1_s1061 + 1.17981 w_k1_ADM_s106
                                 + 1.05039 w_k1_AXP_s106
                                 + 1.04789 w_k1_DIS_s106
                                 + 0.917055 w_k1_FDX_s106
                                 + 1.22843 w_k1_IBM_s106
                                 + 1.00771 w_k1_INTC_s106
                                 + 1.11795 w_k1_KO_s106 + 1.00253 w_k1_MAT_s106
                                 + 0.996611 w_k1_MSFT_s106
                                 + 0.950321 w_k1_PG_s106 >= 1
 z_k1_s1062_constraint:          var_1 + z_k1_s1062 + 1.18078 w_k1_ADM_s106
                                 + 0.9846 w_k1_AXP_s106 + 0.96973 w_k1_DIS_s106
                                 + 1.04274 w_k1_FDX_s106
                                 + 0.942703 w_k1_IBM_s106
                                 + 0.987155 w_k1_INTC_s106
                                 + 0.959216 w_k1_KO_s106
                                 + 0.820794 w_k1_MAT_s106
                                 + 1.07657 w_k1_MSFT_s106
                                 + 1.01183 w_k1_PG_s106 >= 1
 z_k1_s1063_constraint:          var_1 + z_k1_s1063 + 0.965323 w_k1_ADM_s106
                                 + 0.777588 w_k1_AXP_s106
                                 + 0.761692 w_k1_DIS_s106
                                 + 0.942755 w_k1_FDX_s106
                                 + 0.740035 w_k1_IBM_s106
                                 + 0.815641 w_k1_INTC_s106
                                 + 0.718649 w_k1_KO_s106
                                 + 0.651801 w_k1_MAT_s106
                                 + 0.907893 w_k1_MSFT_s106
                                 + 0.883794 w_k1_PG_s106 >= 1
 z_k1_s1064_constraint:          var_1 + z_k1_s1064 + 1.2991 w_k1_ADM_s106
                                 + 1.29351 w_k1_AXP_s106
                                 + 1.43639 w_k1_DIS_s106
                                 + 1.26831 w_k1_FDX_s106
                                 + 1.24741 w_k1_IBM_s106
                                 + 1.08205 w_k1_INTC_s106
                                 + 1.34755 w_k1_KO_s106 + 1.27328 w_k1_MAT_s106
                                 + 1.21297 w_k1_MSFT_s106
                                 + 1.17358 w_k1_PG_s106 >= 1
 z_k1_s1065_constraint:          var_1 + z_k1_s1065 + 1.32916 w_k1_ADM_s106
                                 + 1.06619 w_k1_AXP_s106
                                 + 1.27374 w_k1_DIS_s106
                                 + 0.93901 w_k1_FDX_s106
                                 + 1.44823 w_k1_IBM_s106
                                 + 1.00727 w_k1_INTC_s106
                                 + 1.21409 w_k1_KO_s106 + 1.25322 w_k1_MAT_s106
                                 + 0.978825 w_k1_MSFT_s106
                                 + 0.979384 w_k1_PG_s106 >= 1
 z_k1_s1066_constraint:          var_1 + z_k1_s1066 + 0.928595 w_k1_ADM_s106
                                 + 0.755527 w_k1_AXP_s106
                                 + 0.781851 w_k1_DIS_s106
                                 + 0.815427 w_k1_FDX_s106
                                 + 1.07692 w_k1_IBM_s106
                                 + 0.916582 w_k1_INTC_s106
                                 + 0.842892 w_k1_KO_s106
                                 + 0.614299 w_k1_MAT_s106
                                 + 0.80834 w_k1_MSFT_s106
                                 + 0.851865 w_k1_PG_s106 >= 1
 z_k1_s1067_constraint:          var_1 + z_k1_s1067 + 1.09171 w_k1_ADM_s106
                                 + 1.07569 w_k1_AXP_s106
                                 + 1.53862 w_k1_DIS_s106 + 1.2334 w_k1_FDX_s106
                                 + 0.981268 w_k1_IBM_s106
                                 + 1.24396 w_k1_INTC_s106
                                 + 1.05561 w_k1_KO_s106
                                 + 0.748474 w_k1_MAT_s106
                                 + 1.43469 w_k1_MSFT_s106
                                 + 1.07077 w_k1_PG_s106 >= 1
 z_k1_s1070_constraint:          var_1 + z_k1_s1070 + 1.13721 w_k1_ADM_s107
                                 + 1.0884 w_k1_AXP_s107 + 1.6687 w_k1_DIS_s107
                                 + 1.46322 w_k1_FDX_s107
                                 + 0.908968 w_k1_IBM_s107
                                 + 1.38969 w_k1_INTC_s107
                                 + 1.19631 w_k1_KO_s107 + 1.03046 w_k1_MAT_s107
                                 + 1.34997 w_k1_MSFT_s107 + 1.2032 w_k1_PG_s107
                                 >= 1
 z_k1_s1071_constraint:          var_1 + z_k1_s1071 + 0.718307 w_k1_ADM_s107
                                 + 1.05338 w_k1_AXP_s107
                                 + 0.900103 w_k1_DIS_s107
                                 + 0.896933 w_k1_FDX_s107
                                 + 1.68849 w_k1_IBM_s107
                                 + 1.13488 w_k1_INTC_s107
                                 + 1.21095 w_k1_KO_s107 + 1.54859 w_k1_MAT_s107
                                 + 0.918102 w_k1_MSFT_s107
                                 + 1.05315 w_k1_PG_s107 >= 1
 z_k1_s1072_constraint:          var_1 + z_k1_s1072 + 0.832067 w_k1_ADM_s107
                                 + 0.976482 w_k1_AXP_s107
                                 + 1.11516 w_k1_DIS_s107
                                 + 1.03086 w_k1_FDX_s107
                                 + 1.27949 w_k1_IBM_s107
                                 + 1.11207 w_k1_INTC_s107
                                 + 1.21814 w_k1_KO_s107 + 1.3295 w_k1_MAT_s107
                                 + 0.925386 w_k1_MSFT_s107
                                 + 0.998238 w_k1_PG_s107 >= 1
 z_k1_s1073_constraint:          var_1 + z_k1_s1073 + 0.818957 w_k1_ADM_s107
                                 + 0.696712 w_k1_AXP_s107
                                 + 0.725237 w_k1_DIS_s107
                                 + 0.840241 w_k1_FDX_s107
                                 + 0.689918 w_k1_IBM_s107
                                 + 0.863417 w_k1_INTC_s107
                                 + 0.792915 w_k1_KO_s107
                                 + 0.661341 w_k1_MAT_s107
                                 + 0.85489 w_k1_MSFT_s107
                                 + 0.941182 w_k1_PG_s107 >= 1
 z_k1_s1074_constraint:          var_1 + z_k1_s1074 + 1.1365 w_k1_ADM_s107
                                 + 1.15816 w_k1_AXP_s107
                                 + 1.20922 w_k1_DIS_s107 + 1.1907 w_k1_FDX_s107
                                 + 0.889865 w_k1_IBM_s107
                                 + 1.01476 w_k1_INTC_s107 + 1.0382 w_k1_KO_s107
                                 + 1.08425 w_k1_MAT_s107
                                 + 0.990995 w_k1_MSFT_s107
                                 + 1.04625 w_k1_PG_s107 >= 1
 z_k1_s1075_constraint:          var_1 + z_k1_s1075 + 0.879948 w_k1_ADM_s107
                                 + 0.860698 w_k1_AXP_s107
                                 + 1.13251 w_k1_DIS_s107
                                 + 0.985252 w_k1_FDX_s107
                                 + 0.900034 w_k1_IBM_s107
                                 + 1.20225 w_k1_INTC_s107
                                 + 1.03309 w_k1_KO_s107
                                 + 0.922469 w_k1_MAT_s107
                                 + 0.998192 w_k1_MSFT_s107
                                 + 1.00012 w_k1_PG_s107 >= 1
 z_k1_s1076_constraint:          var_1 + z_k1_s1076 + 1.06472 w_k1_ADM_s107
                                 + 1.34622 w_k1_AXP_s107
                                 + 1.52477 w_k1_DIS_s107
                                 + 1.17762 w_k1_FDX_s107
                                 + 1.28577 w_k1_IBM_s107
                                 + 1.27754 w_k1_INTC_s107
                                 + 1.28673 w_k1_KO_s107 + 1.49236 w_k1_MAT_s107
                                 + 1.11235 w_k1_MSFT_s107
                                 + 1.17709 w_k1_PG_s107 >= 1
 z_k1_s1077_constraint:          var_1 + z_k1_s1077 + 0.669919 w_k1_ADM_s107
                                 + 0.685075 w_k1_AXP_s107
                                 + 0.668786 w_k1_DIS_s107
                                 + 0.758174 w_k1_FDX_s107
                                 + 1.09445 w_k1_IBM_s107
                                 + 1.01152 w_k1_INTC_s107
                                 + 0.966714 w_k1_KO_s107
                                 + 1.39509 w_k1_MAT_s107
                                 + 0.828737 w_k1_MSFT_s107
                                 + 0.865383 w_k1_PG_s107 >= 1
 z_k1_s1100_constraint:          var_1 + z_k1_s1100 + 0.8952 w_k1_ADM_s110
                                 + 0.772343 w_k1_AXP_s110
                                 + 0.729407 w_k1_DIS_s110
                                 + 0.880011 w_k1_FDX_s110
                                 + 0.706509 w_k1_IBM_s110
                                 + 0.870907 w_k1_INTC_s110
                                 + 0.744288 w_k1_KO_s110
                                 + 0.648313 w_k1_MAT_s110
                                 + 0.74973 w_k1_MSFT_s110
                                 + 0.882712 w_k1_PG_s110 >= 1
 z_k1_s1101_constraint:          var_1 + z_k1_s1101 + 0.809454 w_k1_ADM_s110
                                 + 0.928347 w_k1_AXP_s110
                                 + 0.846185 w_k1_DIS_s110
                                 + 0.838624 w_k1_FDX_s110
                                 + 1.0188 w_k1_IBM_s110
                                 + 0.873767 w_k1_INTC_s110
                                 + 0.936464 w_k1_KO_s110
                                 + 0.969676 w_k1_MAT_s110
                                 + 0.867185 w_k1_MSFT_s110
                                 + 0.929886 w_k1_PG_s110 >= 1
 z_k1_s1102_constraint:          var_1 + z_k1_s1102 + 1.0882 w_k1_ADM_s110
                                 + 1.13372 w_k1_AXP_s110
                                 + 1.29431 w_k1_DIS_s110
                                 + 1.08869 w_k1_FDX_s110
                                 + 1.18744 w_k1_IBM_s110
                                 + 0.991313 w_k1_INTC_s110
                                 + 1.11778 w_k1_KO_s110 + 1.08856 w_k1_MAT_s110
                                 + 1.00286 w_k1_MSFT_s110
                                 + 1.01248 w_k1_PG_s110 >= 1
 z_k1_s1103_constraint:          var_1 + z_k1_s1103 + 1.20182 w_k1_ADM_s110
                                 + 1.01307 w_k1_AXP_s110
                                 + 1.15158 w_k1_DIS_s110
                                 + 1.12748 w_k1_FDX_s110
                                 + 0.75229 w_k1_IBM_s110
                                 + 0.926516 w_k1_INTC_s110
                                 + 0.963691 w_k1_KO_s110
                                 + 0.801613 w_k1_MAT_s110
                                 + 1.11909 w_k1_MSFT_s110
                                 + 0.973714 w_k1_PG_s110 >= 1
 z_k1_s1104_constraint:          var_1 + z_k1_s1104 + 1.44522 w_k1_ADM_s110
                                 + 1.70914 w_k1_AXP_s110
                                 + 1.66942 w_k1_DIS_s110 + 1.2508 w_k1_FDX_s110
                                 + 1.61823 w_k1_IBM_s110
                                 + 1.01331 w_k1_INTC_s110
                                 + 1.37049 w_k1_KO_s110 + 2.0873 w_k1_MAT_s110
                                 + 1.05267 w_k1_MSFT_s110 + 1.088 w_k1_PG_s110
                                 >= 1
 z_k1_s1105_constraint:          var_1 + z_k1_s1105 + 1.01517 w_k1_ADM_s110
                                 + 1.25902 w_k1_AXP_s110 + 1.0041 w_k1_DIS_s110
                                 + 0.943426 w_k1_FDX_s110
                                 + 1.47439 w_k1_IBM_s110
                                 + 0.928215 w_k1_INTC_s110
                                 + 1.22913 w_k1_KO_s110 + 1.423 w_k1_MAT_s110
                                 + 0.861088 w_k1_MSFT_s110
                                 + 1.10871 w_k1_PG_s110 >= 1
 z_k1_s1106_constraint:          var_1 + z_k1_s1106 + 0.76186 w_k1_ADM_s110
                                 + 0.643679 w_k1_AXP_s110
                                 + 0.560025 w_k1_DIS_s110
                                 + 0.70458 w_k1_FDX_s110
                                 + 0.757189 w_k1_IBM_s110
                                 + 0.774708 w_k1_INTC_s110
                                 + 0.634822 w_k1_KO_s110
                                 + 0.567242 w_k1_MAT_s110
                                 + 0.700441 w_k1_MSFT_s110
                                 + 0.692723 w_k1_PG_s110 >= 1
 z_k1_s1107_constraint:          var_1 + z_k1_s1107 + 1.33843 w_k1_ADM_s110
                                 + 1.53937 w_k1_AXP_s110 + 1.564 w_k1_DIS_s110
                                 + 1.41897 w_k1_FDX_s110 + 1.0328 w_k1_IBM_s110
                                 + 1.21411 w_k1_INTC_s110 + 1.1155 w_k1_KO_s110
                                 + 0.90018 w_k1_MAT_s110
                                 + 1.3029 w_k1_MSFT_s110 + 1.07741 w_k1_PG_s110
                                 >= 1
 z_k1_s1110_constraint:          var_1 + z_k1_s1110 + 1.3572 w_k1_ADM_s111
                                 + 1.51603 w_k1_AXP_s111
                                 + 1.75666 w_k1_DIS_s111
                                 + 1.24041 w_k1_FDX_s111
                                 + 1.63127 w_k1_IBM_s111
                                 + 1.27092 w_k1_INTC_s111
                                 + 1.83125 w_k1_KO_s111 + 1.82281 w_k1_MAT_s111
                                 + 1.57327 w_k1_MSFT_s111
                                 + 1.26225 w_k1_PG_s111 >= 1
 z_k1_s1111_constraint:          var_1 + z_k1_s1111 + 0.927876 w_k1_ADM_s111
                                 + 0.830163 w_k1_AXP_s111
                                 + 1.03603 w_k1_DIS_s111
                                 + 0.901904 w_k1_FDX_s111
                                 + 1.11494 w_k1_IBM_s111
                                 + 1.00972 w_k1_INTC_s111
                                 + 1.08878 w_k1_KO_s111
                                 + 0.884801 w_k1_MAT_s111
                                 + 1.24803 w_k1_MSFT_s111
                                 + 0.86623 w_k1_PG_s111 >= 1
 z_k1_s1112_constraint:          var_1 + z_k1_s1112 + 1.14266 w_k1_ADM_s111
                                 + 1.11194 w_k1_AXP_s111
                                 + 1.04345 w_k1_DIS_s111
                                 + 0.950226 w_k1_FDX_s111
                                 + 1.27217 w_k1_IBM_s111
                                 + 0.845441 w_k1_INTC_s111
                                 + 1.08627 w_k1_KO_s111 + 1.19351 w_k1_MAT_s111
                                 + 1.05198 w_k1_MSFT_s111
                                 + 0.929895 w_k1_PG_s111 >= 1
 z_k1_s1113_constraint:          var_1 + z_k1_s1113 + 1.17366 w_k1_ADM_s111
                                 + 1.12119 w_k1_AXP_s111
                                 + 1.34093 w_k1_DIS_s111
                                 + 1.18542 w_k1_FDX_s111
                                 + 1.03691 w_k1_IBM_s111
                                 + 0.957441 w_k1_INTC_s111
                                 + 1.15978 w_k1_KO_s111 + 1.32436 w_k1_MAT_s111
                                 + 1.22222 w_k1_MSFT_s111
                                 + 1.00496 w_k1_PG_s111 >= 1
 z_k1_s1114_constraint:          var_1 + z_k1_s1114 + 0.951634 w_k1_ADM_s111
                                 + 0.720461 w_k1_AXP_s111
                                 + 0.752406 w_k1_DIS_s111
                                 + 0.752868 w_k1_FDX_s111
                                 + 0.870743 w_k1_IBM_s111
                                 + 0.604393 w_k1_INTC_s111
                                 + 0.827922 w_k1_KO_s111
                                 + 1.04523 w_k1_MAT_s111
                                 + 0.837854 w_k1_MSFT_s111
                                 + 0.829344 w_k1_PG_s111 >= 1
 z_k1_s1115_constraint:          var_1 + z_k1_s1115 + 0.92817 w_k1_ADM_s111
                                 + 0.85003 w_k1_AXP_s111
                                 + 1.11822 w_k1_DIS_s111
                                 + 0.980282 w_k1_FDX_s111
                                 + 0.835045 w_k1_IBM_s111
                                 + 1.09731 w_k1_INTC_s111
                                 + 0.959718 w_k1_KO_s111
                                 + 0.721512 w_k1_MAT_s111
                                 + 1.19227 w_k1_MSFT_s111
                                 + 0.91866 w_k1_PG_s111 >= 1
 z_k1_s1116_constraint:          var_1 + z_k1_s1116 + 1.11843 w_k1_ADM_s111
                                 + 0.937262 w_k1_AXP_s111
                                 + 0.785054 w_k1_DIS_s111
                                 + 0.854702 w_k1_FDX_s111
                                 + 0.953251 w_k1_IBM_s111
                                 + 0.803874 w_k1_INTC_s111
                                 + 0.987422 w_k1_KO_s111
                                 + 0.768413 w_k1_MAT_s111
                                 + 0.861822 w_k1_MSFT_s111
                                 + 0.867989 w_k1_PG_s111 >= 1
 z_k1_s1117_constraint:          var_1 + z_k1_s1117 + 1.01248 w_k1_ADM_s111
                                 + 0.741303 w_k1_AXP_s111
                                 + 0.695126 w_k1_DIS_s111
                                 + 0.789766 w_k1_FDX_s111
                                 + 0.647246 w_k1_IBM_s111
                                 + 0.991885 w_k1_INTC_s111
                                 + 0.741422 w_k1_KO_s111
                                 + 0.461517 w_k1_MAT_s111
                                 + 0.918411 w_k1_MSFT_s111
                                 + 0.828155 w_k1_PG_s111 >= 1
 z_k1_s1120_constraint:          var_1 + z_k1_s1120 + 0.681997 w_k1_ADM_s112
                                 + 0.671555 w_k1_AXP_s112
                                 + 0.656525 w_k1_DIS_s112
                                 + 0.729633 w_k1_FDX_s112
                                 + 0.56249 w_k1_IBM_s112
                                 + 0.823034 w_k1_INTC_s112
                                 + 0.672279 w_k1_KO_s112
                                 + 0.484923 w_k1_MAT_s112
                                 + 0.817051 w_k1_MSFT_s112
                                 + 0.709107 w_k1_PG_s112 >= 1
 z_k1_s1121_constraint:          var_1 + z_k1_s1121 + 0.937361 w_k1_ADM_s112
                                 + 2.00401 w_k1_AXP_s112
                                 + 1.75099 w_k1_DIS_s112
                                 + 1.30356 w_k1_FDX_s112
                                 + 1.33535 w_k1_IBM_s112
                                 + 1.48671 w_k1_INTC_s112
                                 + 1.41092 w_k1_KO_s112 + 1.87125 w_k1_MAT_s112
                                 + 1.20324 w_k1_MSFT_s112
                                 + 1.17733 w_k1_PG_s112 >= 1
 z_k1_s1122_constraint:          var_1 + z_k1_s1122 + 0.817488 w_k1_ADM_s112
                                 + 0.828309 w_k1_AXP_s112
                                 + 0.836102 w_k1_DIS_s112
                                 + 0.822675 w_k1_FDX_s112
                                 + 1.06003 w_k1_IBM_s112
                                 + 1.07102 w_k1_INTC_s112
                                 + 1.05409 w_k1_KO_s112
                                 + 0.845914 w_k1_MAT_s112
                                 + 0.93719 w_k1_MSFT_s112
                                 + 0.971233 w_k1_PG_s112 >= 1
 z_k1_s1123_constraint:          var_1 + z_k1_s1123 + 0.843925 w_k1_ADM_s112
                                 + 1.24184 w_k1_AXP_s112
                                 + 0.978837 w_k1_DIS_s112
                                 + 1.00003 w_k1_FDX_s112
                                 + 1.33313 w_k1_IBM_s112
                                 + 1.07904 w_k1_INTC_s112
                                 + 0.931668 w_k1_KO_s112
                                 + 1.41934 w_k1_MAT_s112
                                 + 0.930398 w_k1_MSFT_s112
                                 + 1.0677 w_k1_PG_s112 >= 1
 z_k1_s1124_constraint:          var_1 + z_k1_s1124 + 0.764269 w_k1_ADM_s112
                                 + 1.0772 w_k1_AXP_s112
                                 + 0.733576 w_k1_DIS_s112
                                 + 0.856299 w_k1_FDX_s112 + 0.716 w_k1_IBM_s112
                                 + 1.02298 w_k1_INTC_s112
                                 + 0.672324 w_k1_KO_s112
                                 + 0.68764 w_k1_MAT_s112
                                 + 0.845918 w_k1_MSFT_s112
                                 + 0.765365 w_k1_PG_s112 >= 1
 z_k1_s1125_constraint:          var_1 + z_k1_s1125 + 1.44607 w_k1_ADM_s112
                                 + 1.77226 w_k1_AXP_s112
                                 + 1.47163 w_k1_DIS_s112
                                 + 1.25661 w_k1_FDX_s112
                                 + 1.68732 w_k1_IBM_s112
                                 + 1.20356 w_k1_INTC_s112
                                 + 1.40946 w_k1_KO_s112 + 2.04045 w_k1_MAT_s112
                                 + 1.23711 w_k1_MSFT_s112
                                 + 1.35368 w_k1_PG_s112 >= 1
 z_k1_s1126_constraint:          var_1 + z_k1_s1126 + 0.816288 w_k1_ADM_s112
                                 + 0.883592 w_k1_AXP_s112
                                 + 0.901232 w_k1_DIS_s112
                                 + 0.859619 w_k1_FDX_s112
                                 + 0.813198 w_k1_IBM_s112
                                 + 0.916482 w_k1_INTC_s112
                                 + 0.838515 w_k1_KO_s112
                                 + 0.67432 w_k1_MAT_s112
                                 + 0.812773 w_k1_MSFT_s112
                                 + 0.875935 w_k1_PG_s112 >= 1
 z_k1_s1127_constraint:          var_1 + z_k1_s1127 + 1.01866 w_k1_ADM_s112
                                 + 1.21668 w_k1_AXP_s112
                                 + 1.48503 w_k1_DIS_s112
                                 + 1.34092 w_k1_FDX_s112
                                 + 1.04304 w_k1_IBM_s112
                                 + 1.19526 w_k1_INTC_s112
                                 + 1.23285 w_k1_KO_s112 + 1.16044 w_k1_MAT_s112
                                 + 1.29062 w_k1_MSFT_s112
                                 + 1.22933 w_k1_PG_s112 >= 1
 z_k1_s1130_constraint:          var_1 + z_k1_s1130 + 0.774727 w_k1_ADM_s113
                                 + 0.703149 w_k1_AXP_s113
                                 + 0.72064 w_k1_DIS_s113
                                 + 0.816314 w_k1_FDX_s113
                                 + 1.0254 w_k1_IBM_s113
                                 + 0.905156 w_k1_INTC_s113
                                 + 1.05217 w_k1_KO_s113 + 1.00988 w_k1_MAT_s113
                                 + 0.785423 w_k1_MSFT_s113
                                 + 0.847483 w_k1_PG_s113 >= 1
 z_k1_s1131_constraint:          var_1 + z_k1_s1131 + 1.02691 w_k1_ADM_s113
                                 + 1.21136 w_k1_AXP_s113
                                 + 0.869165 w_k1_DIS_s113
                                 + 0.925129 w_k1_FDX_s113
                                 + 1.1944 w_k1_IBM_s113
                                 + 1.08995 w_k1_INTC_s113
                                 + 1.16532 w_k1_KO_s113 + 1.48042 w_k1_MAT_s113
                                 + 0.919801 w_k1_MSFT_s113
                                 + 0.980077 w_k1_PG_s113 >= 1
 z_k1_s1132_constraint:          var_1 + z_k1_s1132 + 0.932815 w_k1_ADM_s113
                                 + 0.782026 w_k1_AXP_s113
                                 + 0.813033 w_k1_DIS_s113
                                 + 0.887006 w_k1_FDX_s113
                                 + 0.792323 w_k1_IBM_s113
                                 + 0.81451 w_k1_INTC_s113
                                 + 0.863421 w_k1_KO_s113
                                 + 0.937706 w_k1_MAT_s113
                                 + 0.939293 w_k1_MSFT_s113
                                 + 0.880955 w_k1_PG_s113 >= 1
 z_k1_s1133_constraint:          var_1 + z_k1_s1133 + 1.13397 w_k1_ADM_s113
                                 + 1.24568 w_k1_AXP_s113
                                 + 1.44593 w_k1_DIS_s113
                                 + 1.19617 w_k1_FDX_s113
                                 + 1.69712 w_k1_IBM_s113
                                 + 0.992862 w_k1_INTC_s113
                                 + 1.54974 w_k1_KO_s113 + 2.01423 w_k1_MAT_s113
                                 + 1.15104 w_k1_MSFT_s113
                                 + 1.07028 w_k1_PG_s113 >= 1
 z_k1_s1134_constraint:          var_1 + z_k1_s1134 + 1.35323 w_k1_ADM_s113
                                 + 1.1465 w_k1_AXP_s113 + 1.53063 w_k1_DIS_s113
                                 + 1.43176 w_k1_FDX_s113
                                 + 0.958709 w_k1_IBM_s113
                                 + 1.30254 w_k1_INTC_s113
                                 + 1.55821 w_k1_KO_s113 + 1.37258 w_k1_MAT_s113
                                 + 1.47913 w_k1_MSFT_s113
                                 + 1.22729 w_k1_PG_s113 >= 1
 z_k1_s1135_constraint:          var_1 + z_k1_s1135 + 0.901127 w_k1_ADM_s113
                                 + 0.988463 w_k1_AXP_s113
                                 + 1.15067 w_k1_DIS_s113
                                 + 0.994161 w_k1_FDX_s113
                                 + 1.03141 w_k1_IBM_s113
                                 + 0.906106 w_k1_INTC_s113
                                 + 1.14541 w_k1_KO_s113 + 1.01703 w_k1_MAT_s113
                                 + 1.03828 w_k1_MSFT_s113
                                 + 0.99462 w_k1_PG_s113 >= 1
 z_k1_s1136_constraint:          var_1 + z_k1_s1136 + 1.17934 w_k1_ADM_s113
                                 + 1.24836 w_k1_AXP_s113
                                 + 1.13925 w_k1_DIS_s113
                                 + 1.25254 w_k1_FDX_s113
                                 + 1.00895 w_k1_IBM_s113
                                 + 1.1132 w_k1_INTC_s113 + 1.24661 w_k1_KO_s113
                                 + 1.20315 w_k1_MAT_s113
                                 + 1.05089 w_k1_MSFT_s113
                                 + 1.05369 w_k1_PG_s113 >= 1
 z_k1_s1137_constraint:          var_1 + z_k1_s1137 + 1.06891 w_k1_ADM_s113
                                 + 1.02581 w_k1_AXP_s113
                                 + 1.21105 w_k1_DIS_s113
                                 + 1.02966 w_k1_FDX_s113
                                 + 1.33486 w_k1_IBM_s113
                                 + 1.3493 w_k1_INTC_s113 + 1.44554 w_k1_KO_s113
                                 + 1.19839 w_k1_MAT_s113
                                 + 1.17057 w_k1_MSFT_s113
                                 + 1.15236 w_k1_PG_s113 >= 1
 z_k1_s1140_constraint:          var_1 + z_k1_s1140 + 0.839862 w_k1_ADM_s114
                                 + 0.835994 w_k1_AXP_s114
                                 + 0.73867 w_k1_DIS_s114
                                 + 0.748234 w_k1_FDX_s114
                                 + 0.994182 w_k1_IBM_s114
                                 + 0.733743 w_k1_INTC_s114
                                 + 0.963824 w_k1_KO_s114
                                 + 1.03909 w_k1_MAT_s114
                                 + 0.838815 w_k1_MSFT_s114
                                 + 1.05638 w_k1_PG_s114 >= 1
 z_k1_s1141_constraint:          var_1 + z_k1_s1141 + 0.749196 w_k1_ADM_s114
                                 + 0.922789 w_k1_AXP_s114
                                 + 0.812943 w_k1_DIS_s114
                                 + 0.800536 w_k1_FDX_s114
                                 + 1.42409 w_k1_IBM_s114
                                 + 0.831662 w_k1_INTC_s114
                                 + 1.42239 w_k1_KO_s114 + 1.76276 w_k1_MAT_s114
                                 + 0.932096 w_k1_MSFT_s114
                                 + 1.11062 w_k1_PG_s114 >= 1
 z_k1_s1142_constraint:          var_1 + z_k1_s1142 + 1.3411 w_k1_ADM_s114
                                 + 1.39444 w_k1_AXP_s114
                                 + 1.87224 w_k1_DIS_s114
                                 + 1.37148 w_k1_FDX_s114
                                 + 1.06706 w_k1_IBM_s114
                                 + 1.02461 w_k1_INTC_s114
                                 + 1.39197 w_k1_KO_s114 + 1.48438 w_k1_MAT_s114
                                 + 1.3102 w_k1_MSFT_s114 + 1.23501 w_k1_PG_s114
                                 >= 1
 z_k1_s1143_constraint:          var_1 + z_k1_s1143 + 1.17157 w_k1_ADM_s114
                                 + 1.21474 w_k1_AXP_s114
                                 + 1.32901 w_k1_DIS_s114
                                 + 1.14938 w_k1_FDX_s114
                                 + 1.11013 w_k1_IBM_s114
                                 + 1.01839 w_k1_INTC_s114 + 1.2405 w_k1_KO_s114
                                 + 1.26986 w_k1_MAT_s114
                                 + 1.19009 w_k1_MSFT_s114
                                 + 1.19495 w_k1_PG_s114 >= 1
 z_k1_s1144_constraint:          var_1 + z_k1_s1144 + 1.08364 w_k1_ADM_s114
                                 + 0.99997 w_k1_AXP_s114
                                 + 1.18955 w_k1_DIS_s114
                                 + 1.01807 w_k1_FDX_s114 + 1.0735 w_k1_IBM_s114
                                 + 1.05208 w_k1_INTC_s114 + 1.2312 w_k1_KO_s114
                                 + 0.865336 w_k1_MAT_s114
                                 + 1.09721 w_k1_MSFT_s114 + 1.184 w_k1_PG_s114
                                 >= 1
 z_k1_s1145_constraint:          var_1 + z_k1_s1145 + 0.993546 w_k1_ADM_s114
                                 + 0.771658 w_k1_AXP_s114
                                 + 0.869208 w_k1_DIS_s114
                                 + 0.924927 w_k1_FDX_s114
                                 + 0.693726 w_k1_IBM_s114
                                 + 0.878643 w_k1_INTC_s114
                                 + 0.844155 w_k1_KO_s114
                                 + 0.51919 w_k1_MAT_s114
                                 + 0.999769 w_k1_MSFT_s114
                                 + 0.87814 w_k1_PG_s114 >= 1
 z_k1_s1146_constraint:          var_1 + z_k1_s1146 + 1.4732 w_k1_ADM_s114
                                 + 1.42961 w_k1_AXP_s114
                                 + 1.37977 w_k1_DIS_s114
                                 + 1.21684 w_k1_FDX_s114
                                 + 1.56003 w_k1_IBM_s114
                                 + 1.02783 w_k1_INTC_s114
                                 + 1.43855 w_k1_KO_s114 + 1.70717 w_k1_MAT_s114
                                 + 1.23469 w_k1_MSFT_s114
                                 + 1.34317 w_k1_PG_s114 >= 1
 z_k1_s1147_constraint:          var_1 + z_k1_s1147 + 0.83517 w_k1_ADM_s114
                                 + 0.556085 w_k1_AXP_s114
                                 + 0.789299 w_k1_DIS_s114
                                 + 0.718367 w_k1_FDX_s114
                                 + 1.0538 w_k1_IBM_s114
                                 + 0.670335 w_k1_INTC_s114
                                 + 1.04512 w_k1_KO_s114 + 1.0992 w_k1_MAT_s114
                                 + 0.791067 w_k1_MSFT_s114
                                 + 0.927428 w_k1_PG_s114 >= 1
 z_k1_s1150_constraint:          var_1 + z_k1_s1150 + 0.619411 w_k1_ADM_s115
                                 + 0.713581 w_k1_AXP_s115
                                 + 0.634796 w_k1_DIS_s115
                                 + 0.808185 w_k1_FDX_s115
                                 + 0.701871 w_k1_IBM_s115
                                 + 0.995512 w_k1_INTC_s115
                                 + 0.721951 w_k1_KO_s115
                                 + 0.680358 w_k1_MAT_s115
                                 + 0.837645 w_k1_MSFT_s115
                                 + 0.918246 w_k1_PG_s115 >= 1
 z_k1_s1151_constraint:          var_1 + z_k1_s1151 + 0.831467 w_k1_ADM_s115
                                 + 1.30859 w_k1_AXP_s115
                                 + 0.927322 w_k1_DIS_s115
                                 + 1.09792 w_k1_FDX_s115
                                 + 0.900232 w_k1_IBM_s115
                                 + 1.19346 w_k1_INTC_s115
                                 + 1.01706 w_k1_KO_s115
                                 + 0.895674 w_k1_MAT_s115
                                 + 0.873439 w_k1_MSFT_s115
                                 + 1.04899 w_k1_PG_s115 >= 1
 z_k1_s1152_constraint:          var_1 + z_k1_s1152 + 0.725739 w_k1_ADM_s115
                                 + 1.07266 w_k1_AXP_s115
                                 + 0.902387 w_k1_DIS_s115
                                 + 0.848359 w_k1_FDX_s115
                                 + 1.29932 w_k1_IBM_s115
                                 + 1.09359 w_k1_INTC_s115
                                 + 1.23775 w_k1_KO_s115 + 1.44006 w_k1_MAT_s115
                                 + 0.953858 w_k1_MSFT_s115
                                 + 1.0189 w_k1_PG_s115 >= 1
 z_k1_s1153_constraint:          var_1 + z_k1_s1153 + 1.01028 w_k1_ADM_s115
                                 + 1.06021 w_k1_AXP_s115
                                 + 1.06287 w_k1_DIS_s115
                                 + 1.11805 w_k1_FDX_s115
                                 + 0.674117 w_k1_IBM_s115
                                 + 0.945217 w_k1_INTC_s115
                                 + 0.749548 w_k1_KO_s115
                                 + 0.526036 w_k1_MAT_s115
                                 + 1.04605 w_k1_MSFT_s115
                                 + 1.01533 w_k1_PG_s115 >= 1
 z_k1_s1154_constraint:          var_1 + z_k1_s1154 + 0.904847 w_k1_ADM_s115
                                 + 1.81755 w_k1_AXP_s115
                                 + 1.71638 w_k1_DIS_s115
                                 + 1.43406 w_k1_FDX_s115 + 1.4136 w_k1_IBM_s115
                                 + 1.59449 w_k1_INTC_s115
                                 + 1.60887 w_k1_KO_s115 + 1.56721 w_k1_MAT_s115
                                 + 1.44128 w_k1_MSFT_s115
                                 + 1.35142 w_k1_PG_s115 >= 1
 z_k1_s1155_constraint:          var_1 + z_k1_s1155 + 0.698827 w_k1_ADM_s115
                                 + 1.29302 w_k1_AXP_s115
                                 + 1.27436 w_k1_DIS_s115
                                 + 0.977314 w_k1_FDX_s115
                                 + 1.91718 w_k1_IBM_s115
                                 + 1.45821 w_k1_INTC_s115
                                 + 1.41067 w_k1_KO_s115 + 2.19627 w_k1_MAT_s115
                                 + 1.30683 w_k1_MSFT_s115
                                 + 1.04267 w_k1_PG_s115 >= 1
 z_k1_s1156_constraint:          var_1 + z_k1_s1156 + 0.717945 w_k1_ADM_s115
                                 + 1.21524 w_k1_AXP_s115
                                 + 0.813445 w_k1_DIS_s115
                                 + 0.963759 w_k1_FDX_s115
                                 + 0.96744 w_k1_IBM_s115
                                 + 1.20446 w_k1_INTC_s115
                                 + 0.96594 w_k1_KO_s115 + 1.22013 w_k1_MAT_s115
                                 + 0.832427 w_k1_MSFT_s115
                                 + 1.00217 w_k1_PG_s115 >= 1
 z_k1_s1157_constraint:          var_1 + z_k1_s1157 + 0.734944 w_k1_ADM_s115
                                 + 1.12394 w_k1_AXP_s115
                                 + 1.40777 w_k1_DIS_s115
                                 + 1.12272 w_k1_FDX_s115
                                 + 0.954374 w_k1_IBM_s115
                                 + 1.4554 w_k1_INTC_s115 + 1.05257 w_k1_KO_s115
                                 + 0.812034 w_k1_MAT_s115
                                 + 1.2688 w_k1_MSFT_s115 + 1.00553 w_k1_PG_s115
                                 >= 1
 z_k1_s1160_constraint:          var_1 + z_k1_s1160 + 1.25236 w_k1_ADM_s116
                                 + 1.14847 w_k1_AXP_s116
                                 + 1.21948 w_k1_DIS_s116
                                 + 1.15082 w_k1_FDX_s116
                                 + 0.952279 w_k1_IBM_s116
                                 + 1.23839 w_k1_INTC_s116
                                 + 0.922252 w_k1_KO_s116
                                 + 0.702258 w_k1_MAT_s116
                                 + 1.07521 w_k1_MSFT_s116
                                 + 0.938553 w_k1_PG_s116 >= 1
 z_k1_s1161_constraint:          var_1 + z_k1_s1161 + 0.837309 w_k1_ADM_s116
                                 + 0.739995 w_k1_AXP_s116
                                 + 0.769748 w_k1_DIS_s116
                                 + 0.946095 w_k1_FDX_s116
                                 + 0.561092 w_k1_IBM_s116
                                 + 1.06364 w_k1_INTC_s116
                                 + 0.714546 w_k1_KO_s116
                                 + 0.32213 w_k1_MAT_s116
                                 + 1.0186 w_k1_MSFT_s116
                                 + 0.857312 w_k1_PG_s116 >= 1
 z_k1_s1162_constraint:          var_1 + z_k1_s1162 + 1.30013 w_k1_ADM_s116
                                 + 1.6552 w_k1_AXP_s116 + 1.63951 w_k1_DIS_s116
                                 + 1.35443 w_k1_FDX_s116
                                 + 1.19806 w_k1_IBM_s116
                                 + 1.08301 w_k1_INTC_s116
                                 + 1.05822 w_k1_KO_s116 + 1.10308 w_k1_MAT_s116
                                 + 1.15958 w_k1_MSFT_s116 + 1.0245 w_k1_PG_s116
                                 >= 1
 z_k1_s1163_constraint:          var_1 + z_k1_s1163 + 0.998195 w_k1_ADM_s116
                                 + 1.03171 w_k1_AXP_s116
                                 + 1.15376 w_k1_DIS_s116
                                 + 1.04413 w_k1_FDX_s116
                                 + 1.11372 w_k1_IBM_s116
                                 + 0.983972 w_k1_INTC_s116
                                 + 1.02774 w_k1_KO_s116 + 1.00109 w_k1_MAT_s116
                                 + 1.11259 w_k1_MSFT_s116
                                 + 1.00325 w_k1_PG_s116 >= 1
 z_k1_s1164_constraint:          var_1 + z_k1_s1164 + 1.07565 w_k1_ADM_s116
                                 + 1.12599 w_k1_AXP_s116
                                 + 1.13178 w_k1_DIS_s116
                                 + 1.12362 w_k1_FDX_s116
                                 + 1.51969 w_k1_IBM_s116
                                 + 0.987969 w_k1_INTC_s116
                                 + 1.00896 w_k1_KO_s116 + 1.27324 w_k1_MAT_s116
                                 + 1.08893 w_k1_MSFT_s116
                                 + 1.08019 w_k1_PG_s116 >= 1
 z_k1_s1165_constraint:          var_1 + z_k1_s1165 + 1.08433 w_k1_ADM_s116
                                 + 1.09415 w_k1_AXP_s116
                                 + 0.919261 w_k1_DIS_s116
                                 + 0.981155 w_k1_FDX_s116
                                 + 0.816619 w_k1_IBM_s116
                                 + 0.967413 w_k1_INTC_s116
                                 + 0.818233 w_k1_KO_s116 + 0.6815 w_k1_MAT_s116
                                 + 0.9118 w_k1_MSFT_s116 + 1.0426 w_k1_PG_s116
                                 >= 1
 z_k1_s1166_constraint:          var_1 + z_k1_s1166 + 0.972557 w_k1_ADM_s116
                                 + 0.603286 w_k1_AXP_s116
                                 + 0.671254 w_k1_DIS_s116
                                 + 0.76651 w_k1_FDX_s116
                                 + 0.805993 w_k1_IBM_s116
                                 + 0.829752 w_k1_INTC_s116
                                 + 0.647564 w_k1_KO_s116 + 0.4739 w_k1_MAT_s116
                                 + 0.829771 w_k1_MSFT_s116
                                 + 0.765058 w_k1_PG_s116 >= 1
 z_k1_s1167_constraint:          var_1 + z_k1_s1167 + 1.07738 w_k1_ADM_s116
                                 + 0.956476 w_k1_AXP_s116
                                 + 0.797738 w_k1_DIS_s116
                                 + 0.846268 w_k1_FDX_s116
                                 + 1.1013 w_k1_IBM_s116
                                 + 0.971395 w_k1_INTC_s116
                                 + 0.921502 w_k1_KO_s116
                                 + 0.78873 w_k1_MAT_s116
                                 + 0.96676 w_k1_MSFT_s116
                                 + 0.911145 w_k1_PG_s116 >= 1
 z_k1_s1170_constraint:          var_1 + z_k1_s1170 + 0.797447 w_k1_ADM_s117
                                 + 1.04564 w_k1_AXP_s117
                                 + 0.964881 w_k1_DIS_s117
                                 + 0.876426 w_k1_FDX_s117
                                 + 1.60039 w_k1_IBM_s117
                                 + 0.936563 w_k1_INTC_s117
                                 + 1.42365 w_k1_KO_s117 + 1.69608 w_k1_MAT_s117
                                 + 1.02072 w_k1_MSFT_s117
                                 + 1.06716 w_k1_PG_s117 >= 1
 z_k1_s1171_constraint:          var_1 + z_k1_s1171 + 0.59066 w_k1_ADM_s117
                                 + 0.782827 w_k1_AXP_s117
                                 + 0.576804 w_k1_DIS_s117
                                 + 0.688882 w_k1_FDX_s117
                                 + 0.818259 w_k1_IBM_s117
                                 + 0.756842 w_k1_INTC_s117
                                 + 0.773646 w_k1_KO_s117
                                 + 0.887765 w_k1_MAT_s117
                                 + 0.754088 w_k1_MSFT_s117
                                 + 0.882741 w_k1_PG_s117 >= 1
 z_k1_s1172_constraint:          var_1 + z_k1_s1172 + 0.954674 w_k1_ADM_s117
                                 + 1.23946 w_k1_AXP_s117 + 1.4839 w_k1_DIS_s117
                                 + 1.15321 w_k1_FDX_s117
                                 + 0.928999 w_k1_IBM_s117
                                 + 1.05523 w_k1_INTC_s117
                                 + 1.23529 w_k1_KO_s117 + 1.18736 w_k1_MAT_s117
                                 + 1.39898 w_k1_MSFT_s117
                                 + 1.19813 w_k1_PG_s117 >= 1
 z_k1_s1173_constraint:          var_1 + z_k1_s1173 + 0.526077 w_k1_ADM_s117
                                 + 0.626529 w_k1_AXP_s117
                                 + 0.794255 w_k1_DIS_s117
                                 + 0.843512 w_k1_FDX_s117
                                 + 0.706311 w_k1_IBM_s117
                                 + 0.93459 w_k1_INTC_s117
                                 + 0.821494 w_k1_KO_s117
                                 + 0.657704 w_k1_MAT_s117
                                 + 0.98122 w_k1_MSFT_s117
                                 + 0.948382 w_k1_PG_s117 >= 1
 z_k1_s1174_constraint:          var_1 + z_k1_s1174 + 0.849019 w_k1_ADM_s117
                                 + 0.870133 w_k1_AXP_s117
                                 + 0.900221 w_k1_DIS_s117
                                 + 0.976865 w_k1_FDX_s117
                                 + 0.817883 w_k1_IBM_s117
                                 + 1.04731 w_k1_INTC_s117
                                 + 1.01678 w_k1_KO_s117 + 1.22578 w_k1_MAT_s117
                                 + 1.14308 w_k1_MSFT_s117
                                 + 1.09544 w_k1_PG_s117 >= 1
 z_k1_s1175_constraint:          var_1 + z_k1_s1175 + 0.861637 w_k1_ADM_s117
                                 + 1.26044 w_k1_AXP_s117
                                 + 1.61641 w_k1_DIS_s117
                                 + 1.30185 w_k1_FDX_s117
                                 + 1.37764 w_k1_IBM_s117
                                 + 0.977293 w_k1_INTC_s117
                                 + 1.52736 w_k1_KO_s117 + 1.47832 w_k1_MAT_s117
                                 + 1.3241 w_k1_MSFT_s117 + 1.18086 w_k1_PG_s117
                                 >= 1
 z_k1_s1176_constraint:          var_1 + z_k1_s1176 + 0.776165 w_k1_ADM_s117
                                 + 0.775576 w_k1_AXP_s117
                                 + 0.761885 w_k1_DIS_s117
                                 + 0.798833 w_k1_FDX_s117
                                 + 1.01305 w_k1_IBM_s117
                                 + 0.686814 w_k1_INTC_s117
                                 + 0.88518 w_k1_KO_s117
                                 + 0.988023 w_k1_MAT_s117
                                 + 0.879262 w_k1_MSFT_s117
                                 + 0.893037 w_k1_PG_s117 >= 1
 z_k1_s1177_constraint:          var_1 + z_k1_s1177 + 0.78489 w_k1_ADM_s117
                                 + 1.08781 w_k1_AXP_s117
                                 + 0.984609 w_k1_DIS_s117
                                 + 0.961278 w_k1_FDX_s117
                                 + 1.18539 w_k1_IBM_s117
                                 + 0.758695 w_k1_INTC_s117
                                 + 0.935457 w_k1_KO_s117
                                 + 1.18638 w_k1_MAT_s117
                                 + 1.06289 w_k1_MSFT_s117
                                 + 1.05943 w_k1_PG_s117 >= 1
 k1_return_constraint:           0.0134735938532736 w_k1_ADM_s000
                                 + 0.0206503027980798 w_k1_ADM_s001
                                 + 0.0372648233937535 w_k1_ADM_s002
                                 + 0.0117696091663587 w_k1_ADM_s003
                                 + 0.0514577560794515 w_k1_ADM_s004
                                 + 0.0206405932511086 w_k1_ADM_s005
                                 + 0.0198365785978183 w_k1_ADM_s006
                                 + 0.0276991519782913 w_k1_ADM_s007
                                 + 0.0651478846704442 w_k1_ADM_s010
                                 + 0.0689312533525988 w_k1_ADM_s011
                                 + 0.0176088587253753 w_k1_ADM_s012
                                 + 0.0163280957302612 w_k1_ADM_s013
                                 + 0.0308251468216588 w_k1_ADM_s014
                                 + 0.0177157973693457 w_k1_ADM_s015
                                 + 0.039555937888221 w_k1_ADM_s016
                                 + 0.0279491316569754 w_k1_ADM_s017
                                 + 0.0500872650658718 w_k1_ADM_s100
                                 + 0.0311779602096157 w_k1_ADM_s101
                                 + 0.0155853919855358 w_k1_ADM_s102
                                 + 0.0436164244007151 w_k1_ADM_s103
                                 + 0.0842863044605522 w_k1_ADM_s104
                                 + 0.0206083847489797 w_k1_ADM_s105
                                 + 0.0209571113313422 w_k1_ADM_s106
                                 + 0.0707893730786736 w_k1_ADM_s107
                                 + 0.0450809496258677 w_k1_ADM_s110
                                 + 0.0217058684172891 w_k1_ADM_s111
                                 + 0.0172490967435236 w_k1_ADM_s112
                                 + 0.0126689193689923 w_k1_ADM_s113
                                 + 0.0277077354968329 w_k1_ADM_s114
                                 + 0.0295492840936244 w_k1_ADM_s115
                                 + 0.0255706365093939 w_k1_ADM_s116
                                 + 0.00822411314975328 w_k1_ADM_s117
                                 + 0.0131180562235508 w_k1_AXP_s000
                                 + 0.0238572788717422 w_k1_AXP_s001
                                 + 0.0375686943248735 w_k1_AXP_s002
                                 + 0.0125793340989457 w_k1_AXP_s003
                                 + 0.0501105769020049 w_k1_AXP_s004
                                 + 0.0227540807208389 w_k1_AXP_s005
                                 + 0.0179597345808104 w_k1_AXP_s006
                                 + 0.0275421369183688 w_k1_AXP_s007
                                 + 0.0609625366165444 w_k1_AXP_s010
                                 + 0.069650862304764 w_k1_AXP_s011
                                 + 0.0149652923873445 w_k1_AXP_s012
                                 + 0.0135716231250456 w_k1_AXP_s013
                                 + 0.0279457271279438 w_k1_AXP_s014
                                 + 0.0178290422180283 w_k1_AXP_s015
                                 + 0.0415299351826183 w_k1_AXP_s016
                                 + 0.0307853321074323 w_k1_AXP_s017
                                 + 0.0507330064438171 w_k1_AXP_s100
                                 + 0.0304791820335649 w_k1_AXP_s101
                                 + 0.01725713858656 w_k1_AXP_s102
                                 + 0.0476888336167384 w_k1_AXP_s103
                                 + 0.080482275919608 w_k1_AXP_s104
                                 + 0.0205269647092037 w_k1_AXP_s105
                                 + 0.0184022465866231 w_k1_AXP_s106
                                 + 0.0747206318224719 w_k1_AXP_s107
                                 + 0.0464175773044975 w_k1_AXP_s110
                                 + 0.019882043995363 w_k1_AXP_s111
                                 + 0.0213896703456395 w_k1_AXP_s112
                                 + 0.0122868703554915 w_k1_AXP_s113
                                 + 0.0252826887368859 w_k1_AXP_s114
                                 + 0.0432167003541896 w_k1_AXP_s115
                                 + 0.0242516064314391 w_k1_AXP_s116
                                 + 0.0102965893874189 w_k1_AXP_s117
                                 + 0.0138097742283075 w_k1_DIS_s000
                                 + 0.0240550244814277 w_k1_DIS_s001
                                 + 0.0332550719482164 w_k1_DIS_s002
                                 + 0.0131429198164292 w_k1_DIS_s003
                                 + 0.0489457713851249 w_k1_DIS_s004
                                 + 0.0222398379786334 w_k1_DIS_s005
                                 + 0.0182904199589332 w_k1_DIS_s006
                                 + 0.0260096672141547 w_k1_DIS_s007
                                 + 0.0703720832947859 w_k1_DIS_s010
                                 + 0.0696958698234844 w_k1_DIS_s011
                                 + 0.0160772202688945 w_k1_DIS_s012
                                 + 0.0156675615596987 w_k1_DIS_s013
                                 + 0.0288465462639763 w_k1_DIS_s014
                                 + 0.0156960858238312 w_k1_DIS_s015
                                 + 0.0389149181809272 w_k1_DIS_s016
                                 + 0.0315102826269532 w_k1_DIS_s017
                                 + 0.0509591486376251 w_k1_DIS_s100
                                 + 0.035183797584807 w_k1_DIS_s101
                                 + 0.0175515643453936 w_k1_DIS_s102
                                 + 0.0452843244849293 w_k1_DIS_s103
                                 + 0.0812467892565413 w_k1_DIS_s104
                                 + 0.0206383273134308 w_k1_DIS_s105
                                 + 0.0204914624584483 w_k1_DIS_s106
                                 + 0.0847173090931916 w_k1_DIS_s107
                                 + 0.0464139997394829 w_k1_DIS_s110
                                 + 0.0214557324421952 w_k1_DIS_s111
                                 + 0.0208175209526146 w_k1_DIS_s112
                                 + 0.0128190201405989 w_k1_DIS_s113
                                 + 0.0287121859725817 w_k1_DIS_s114
                                 + 0.0395874326389577 w_k1_DIS_s115
                                 + 0.0244537515942009 w_k1_DIS_s116
                                 + 0.0108270735322931 w_k1_DIS_s117
                                 + 0.0130885830484981 w_k1_FDX_s000
                                 + 0.0240966405765637 w_k1_FDX_s001
                                 + 0.0352876314856047 w_k1_FDX_s002
                                 + 0.0124608643381131 w_k1_FDX_s003
                                 + 0.0471683238860125 w_k1_FDX_s004
                                 + 0.0211186855685926 w_k1_FDX_s005
                                 + 0.0177988598213017 w_k1_FDX_s006
                                 + 0.0260216859504978 w_k1_FDX_s007
                                 + 0.064058736917073 w_k1_FDX_s010
                                 + 0.0676027334223764 w_k1_FDX_s011
                                 + 0.0149563765182534 w_k1_FDX_s012
                                 + 0.0152995216964662 w_k1_FDX_s013
                                 + 0.0288559504489114 w_k1_FDX_s014
                                 + 0.0152904414576035 w_k1_FDX_s015
                                 + 0.0397357531268373 w_k1_FDX_s016
                                 + 0.0333740703734317 w_k1_FDX_s017
                                 + 0.0501934985106128 w_k1_FDX_s100
                                 + 0.0335097600742724 w_k1_FDX_s101
                                 + 0.0171433985785743 w_k1_FDX_s102
                                 + 0.0471535341093944 w_k1_FDX_s103
                                 + 0.08032950011439 w_k1_FDX_s104
                                 + 0.0202650067450405 w_k1_FDX_s105
                                 + 0.0193718145405733 w_k1_FDX_s106
                                 + 0.0804539484450452 w_k1_FDX_s107
                                 + 0.043995781858012 w_k1_FDX_s110
                                 + 0.0194150438306538 w_k1_FDX_s111
                                 + 0.0195435285816478 w_k1_FDX_s112
                                 + 0.0126746654538912 w_k1_FDX_s113
                                 + 0.0260059099284221 w_k1_FDX_s114
                                 + 0.0390684780520609 w_k1_FDX_s115
                                 + 0.0242934076004549 w_k1_FDX_s116
                                 + 0.0102110284685403 w_k1_FDX_s117
                                 + 0.0132830009326043 w_k1_IBM_s000
                                 + 0.0244870594102586 w_k1_IBM_s001
                                 + 0.0330870895458642 w_k1_IBM_s002
                                 + 0.0122967005914569 w_k1_IBM_s003
                                 + 0.0468507278316928 w_k1_IBM_s004
                                 + 0.0228461771484715 w_k1_IBM_s005
                                 + 0.018368635448676 w_k1_IBM_s006
                                 + 0.0272331635175902 w_k1_IBM_s007
                                 + 0.064119594686043 w_k1_IBM_s010
                                 + 0.0638364236704611 w_k1_IBM_s011
                                 + 0.0145892698095831 w_k1_IBM_s012
                                 + 0.0157972652236241 w_k1_IBM_s013
                                 + 0.0278458741597125 w_k1_IBM_s014
                                 + 0.0160781985265086 w_k1_IBM_s015
                                 + 0.040792600404818 w_k1_IBM_s016
                                 + 0.0333449763531032 w_k1_IBM_s017
                                 + 0.051456901327741 w_k1_IBM_s100
                                 + 0.0339637282423697 w_k1_IBM_s101
                                 + 0.0174232466469329 w_k1_IBM_s102
                                 + 0.0487576712966418 w_k1_IBM_s103
                                 + 0.0827947988852144 w_k1_IBM_s104
                                 + 0.0195613855312341 w_k1_IBM_s105
                                 + 0.0196458634945315 w_k1_IBM_s106
                                 + 0.0815053132778589 w_k1_IBM_s107
                                 + 0.0441499157043743 w_k1_IBM_s110
                                 + 0.0208704470473566 w_k1_IBM_s111
                                 + 0.0192196429017829 w_k1_IBM_s112
                                 + 0.0131355943014965 w_k1_IBM_s113
                                 + 0.0265461869382823 w_k1_IBM_s114
                                 + 0.039426796098517 w_k1_IBM_s115
                                 + 0.025275451487325 w_k1_IBM_s116
                                 + 0.0110137322988124 w_k1_IBM_s117
                                 + 0.0122607799411001 w_k1_INTC_s000
                                 + 0.024199313038949 w_k1_INTC_s001
                                 + 0.0346126589477411 w_k1_INTC_s002
                                 + 0.0110697860111925 w_k1_INTC_s003
                                 + 0.0450028914335831 w_k1_INTC_s004
                                 + 0.0208871698703724 w_k1_INTC_s005
                                 + 0.0161735642148804 w_k1_INTC_s006
                                 + 0.0271973478294839 w_k1_INTC_s007
                                 + 0.0633613256014588 w_k1_INTC_s010
                                 + 0.0633026402382923 w_k1_INTC_s011
                                 + 0.0139598911407201 w_k1_INTC_s012
                                 + 0.0167975638957623 w_k1_INTC_s013
                                 + 0.0254970677594072 w_k1_INTC_s014
                                 + 0.0139931837239065 w_k1_INTC_s015
                                 + 0.0389238858797541 w_k1_INTC_s016
                                 + 0.0347837277790714 w_k1_INTC_s017
                                 + 0.042863308289006 w_k1_INTC_s100
                                 + 0.0288766742279897 w_k1_INTC_s101
                                 + 0.016577775808385 w_k1_INTC_s102
                                 + 0.0488237666357177 w_k1_INTC_s103
                                 + 0.0860936122637554 w_k1_INTC_s104
                                 + 0.0201379671686965 w_k1_INTC_s105
                                 + 0.0189828525566582 w_k1_INTC_s106
                                 + 0.0854924470588472 w_k1_INTC_s107
                                 + 0.0407963803958539 w_k1_INTC_s110
                                 + 0.0182878808077886 w_k1_INTC_s111
                                 + 0.0206046221848831 w_k1_INTC_s112
                                 + 0.0125003437787227 w_k1_INTC_s113
                                 + 0.0239206085503995 w_k1_INTC_s114
                                 + 0.0457993558289542 w_k1_INTC_s115
                                 + 0.0236837615143033 w_k1_INTC_s116
                                 + 0.00941114039448707 w_k1_INTC_s117
                                 + 0.0136687126944255 w_k1_KO_s000
                                 + 0.0229980838643307 w_k1_KO_s001
                                 + 0.0332356284496282 w_k1_KO_s002
                                 + 0.0124197708639549 w_k1_KO_s003
                                 + 0.0481960013352891 w_k1_KO_s004
                                 + 0.0233904170784774 w_k1_KO_s005
                                 + 0.0173490097050033 w_k1_KO_s006
                                 + 0.0246867739885585 w_k1_KO_s007
                                 + 0.0629558679677208 w_k1_KO_s010
                                 + 0.0651619319874842 w_k1_KO_s011
                                 + 0.0150146967691853 w_k1_KO_s012
                                 + 0.0159761232015136 w_k1_KO_s013
                                 + 0.0274752160474886 w_k1_KO_s014
                                 + 0.0153258320828061 w_k1_KO_s015
                                 + 0.0379650765946724 w_k1_KO_s016
                                 + 0.0305661338585721 w_k1_KO_s017
                                 + 0.0470703962449696 w_k1_KO_s100
                                 + 0.0318902252061155 w_k1_KO_s101
                                 + 0.0169193790295929 w_k1_KO_s102
                                 + 0.0494077069150702 w_k1_KO_s103
                                 + 0.0807208296851587 w_k1_KO_s104
                                 + 0.0202175310342008 w_k1_KO_s105
                                 + 0.0188536117106761 w_k1_KO_s106
                                 + 0.0834519624701334 w_k1_KO_s107
                                 + 0.0428886954483709 w_k1_KO_s110
                                 + 0.0213986929685121 w_k1_KO_s111
                                 + 0.0193722873158335 w_k1_KO_s112
                                 + 0.0143907788868089 w_k1_KO_s113
                                 + 0.0293994512204379 w_k1_KO_s114
                                 + 0.0402228713407667 w_k1_KO_s115
                                 + 0.0212087797694949 w_k1_KO_s116
                                 + 0.0112194066123875 w_k1_KO_s117
                                 + 0.0130860596948366 w_k1_MAT_s000
                                 + 0.0245756334675403 w_k1_MAT_s001
                                 + 0.0420084234038568 w_k1_MAT_s002
                                 + 0.0136342639213226 w_k1_MAT_s003
                                 + 0.0455519354138109 w_k1_MAT_s004
                                 + 0.0264489476567213 w_k1_MAT_s005
                                 + 0.0222845429113632 w_k1_MAT_s006
                                 + 0.0294893739133309 w_k1_MAT_s007
                                 + 0.0705695797597803 w_k1_MAT_s010
                                 + 0.0660419390577466 w_k1_MAT_s011
                                 + 0.0154943310610148 w_k1_MAT_s012
                                 + 0.0156992259768485 w_k1_MAT_s013
                                 + 0.0264512381360986 w_k1_MAT_s014
                                 + 0.0152174210670221 w_k1_MAT_s015
                                 + 0.0411245604657624 w_k1_MAT_s016
                                 + 0.0329001516358274 w_k1_MAT_s017
                                 + 0.0555801164120616 w_k1_MAT_s100
                                 + 0.0317279518601446 w_k1_MAT_s101
                                 + 0.0185770118883949 w_k1_MAT_s102
                                 + 0.0635975979911598 w_k1_MAT_s103
                                 + 0.0761350076042497 w_k1_MAT_s104
                                 + 0.0163022939929937 w_k1_MAT_s105
                                 + 0.0157977774280846 w_k1_MAT_s106
                                 + 0.0876019576923747 w_k1_MAT_s107
                                 + 0.0433590049435975 w_k1_MAT_s110
                                 + 0.0213081388621662 w_k1_MAT_s111
                                 + 0.0196232677753009 w_k1_MAT_s112
                                 + 0.0148836818453032 w_k1_MAT_s113
                                 + 0.0270500593612138 w_k1_MAT_s114
                                 + 0.0407934053412021 w_k1_MAT_s115
                                 + 0.0198948652717087 w_k1_MAT_s116
                                 + 0.0121763494315436 w_k1_MAT_s117
                                 + 0.0135251116063228 w_k1_MSFT_s000
                                 + 0.0213851271945528 w_k1_MSFT_s001
                                 + 0.0349528816930015 w_k1_MSFT_s002
                                 + 0.0120811119665263 w_k1_MSFT_s003
                                 + 0.0488075255893917 w_k1_MSFT_s004
                                 + 0.0213705275804149 w_k1_MSFT_s005
                                 + 0.0170705536199873 w_k1_MSFT_s006
                                 + 0.0278342838251005 w_k1_MSFT_s007
                                 + 0.0716409287607192 w_k1_MSFT_s010
                                 + 0.066737830124887 w_k1_MSFT_s011
                                 + 0.0141922863740369 w_k1_MSFT_s012
                                 + 0.0160370984548131 w_k1_MSFT_s013
                                 + 0.0286938542682291 w_k1_MSFT_s014
                                 + 0.0142357744877305 w_k1_MSFT_s015
                                 + 0.0406283050820039 w_k1_MSFT_s016
                                 + 0.0312148717458716 w_k1_MSFT_s017
                                 + 0.0449884529230992 w_k1_MSFT_s100
                                 + 0.0317095520615089 w_k1_MSFT_s101
                                 + 0.0171225004967894 w_k1_MSFT_s102
                                 + 0.0510266121120537 w_k1_MSFT_s103
                                 + 0.0758935096449105 w_k1_MSFT_s104
                                 + 0.0189018629071982 w_k1_MSFT_s105
                                 + 0.0201497427681074 w_k1_MSFT_s106
                                 + 0.0766455776584423 w_k1_MSFT_s107
                                 + 0.0412745879220817 w_k1_MSFT_s110
                                 + 0.0218388277196623 w_k1_MSFT_s111
                                 + 0.019314182151141 w_k1_MSFT_s112
                                 + 0.0128524045730843 w_k1_MSFT_s113
                                 + 0.0275142959506598 w_k1_MSFT_s114
                                 + 0.0399533290906031 w_k1_MSFT_s115
                                 + 0.0242352103800536 w_k1_MSFT_s116
                                 + 0.0114582103134876 w_k1_MSFT_s117
                                 + 0.0133894365971713 w_k1_PG_s000
                                 + 0.0235124320417926 w_k1_PG_s001
                                 + 0.032447549302957 w_k1_PG_s002
                                 + 0.0107786690075301 w_k1_PG_s003
                                 + 0.0481897904473725 w_k1_PG_s004
                                 + 0.0205761128472763 w_k1_PG_s005
                                 + 0.0174747017870997 w_k1_PG_s006
                                 + 0.0251275801379473 w_k1_PG_s007
                                 + 0.0632279627735713 w_k1_PG_s010
                                 + 0.0653908535448527 w_k1_PG_s011
                                 + 0.0133613153088231 w_k1_PG_s012
                                 + 0.0150487188844173 w_k1_PG_s013
                                 + 0.0274127841515054 w_k1_PG_s014
                                 + 0.0148839885807917 w_k1_PG_s015
                                 + 0.041065678508286 w_k1_PG_s016
                                 + 0.0308259136087385 w_k1_PG_s017
                                 + 0.0490583409733686 w_k1_PG_s100
                                 + 0.033534180327044 w_k1_PG_s101
                                 + 0.0162585485549325 w_k1_PG_s102
                                 + 0.048171044923969 w_k1_PG_s103
                                 + 0.0846422065832829 w_k1_PG_s104
                                 + 0.0197883908175885 w_k1_PG_s105
                                 + 0.0186784594245822 w_k1_PG_s106
                                 + 0.0803126259465285 w_k1_PG_s107
                                 + 0.0418129535987322 w_k1_PG_s110
                                 + 0.0186706403330646 w_k1_PG_s111
                                 + 0.019383253430821 w_k1_PG_s112
                                 + 0.0123756779880237 w_k1_PG_s113
                                 + 0.028008950914288 w_k1_PG_s114
                                 + 0.0400156117109228 w_k1_PG_s115
                                 + 0.0228785336392068 w_k1_PG_s116
                                 + 0.0110876450034808 w_k1_PG_s117 >= 1.17
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
