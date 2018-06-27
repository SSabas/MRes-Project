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
 w_k1_ADM_s0_return_constraint:  1.10869 w_0_ADM - w_k1_ADM_s0
                                 + 0.99 b_k1_ADM_s0 - 1.01 s_k1_ADM_s0  = 0
 w_k1_ADM_s1_return_constraint:  0.933695 w_0_ADM - w_k1_ADM_s1
                                 + 0.99 b_k1_ADM_s1 - 1.01 s_k1_ADM_s1  = 0
 w_k1_ADM_s00_return_constraint: 0.991323 w_k1_ADM_s0 - w_k1_ADM_s00
                                 + 0.99 b_k1_ADM_s00 - 1.01 s_k1_ADM_s00  = 0
 w_k1_ADM_s01_return_constraint: 0.923194 w_k1_ADM_s0 - w_k1_ADM_s01
                                 + 0.99 b_k1_ADM_s01 - 1.01 s_k1_ADM_s01  = 0
 w_k1_ADM_s10_return_constraint: 0.960748 w_k1_ADM_s1 - w_k1_ADM_s10
                                 + 0.99 b_k1_ADM_s10 - 1.01 s_k1_ADM_s10  = 0
 w_k1_ADM_s11_return_constraint: 1.10015 w_k1_ADM_s1 - w_k1_ADM_s11
                                 + 0.99 b_k1_ADM_s11 - 1.01 s_k1_ADM_s11  = 0
 w_k1_ADM_s000_return_constraint: 1.43115 w_k1_ADM_s00 - w_k1_ADM_s000
                                 + 0.99 b_k1_ADM_s000 - 1.01 s_k1_ADM_s000  = 0
 w_k1_ADM_s001_return_constraint: 0.912553 w_k1_ADM_s00 - w_k1_ADM_s001
                                 + 0.99 b_k1_ADM_s001 - 1.01 s_k1_ADM_s001  = 0
 w_k1_ADM_s002_return_constraint: 1.14071 w_k1_ADM_s00 - w_k1_ADM_s002
                                 + 0.99 b_k1_ADM_s002 - 1.01 s_k1_ADM_s002  = 0
 w_k1_ADM_s003_return_constraint: 0.864778 w_k1_ADM_s00 - w_k1_ADM_s003
                                 + 0.99 b_k1_ADM_s003 - 1.01 s_k1_ADM_s003  = 0
 w_k1_ADM_s004_return_constraint: 1.22254 w_k1_ADM_s00 - w_k1_ADM_s004
                                 + 0.99 b_k1_ADM_s004 - 1.01 s_k1_ADM_s004  = 0
 w_k1_ADM_s005_return_constraint: 1.19012 w_k1_ADM_s00 - w_k1_ADM_s005
                                 + 0.99 b_k1_ADM_s005 - 1.01 s_k1_ADM_s005  = 0
 w_k1_ADM_s006_return_constraint: 1.20161 w_k1_ADM_s00 - w_k1_ADM_s006
                                 + 0.99 b_k1_ADM_s006 - 1.01 s_k1_ADM_s006  = 0
 w_k1_ADM_s007_return_constraint: 1.03061 w_k1_ADM_s00 - w_k1_ADM_s007
                                 + 0.99 b_k1_ADM_s007 - 1.01 s_k1_ADM_s007  = 0
 w_k1_ADM_s010_return_constraint: 0.962022 w_k1_ADM_s01 - w_k1_ADM_s010
                                 + 0.99 b_k1_ADM_s010 - 1.01 s_k1_ADM_s010  = 0
 w_k1_ADM_s011_return_constraint: 0.789618 w_k1_ADM_s01 - w_k1_ADM_s011
                                 + 0.99 b_k1_ADM_s011 - 1.01 s_k1_ADM_s011  = 0
 w_k1_ADM_s012_return_constraint: 0.976253 w_k1_ADM_s01 - w_k1_ADM_s012
                                 + 0.99 b_k1_ADM_s012 - 1.01 s_k1_ADM_s012  = 0
 w_k1_ADM_s013_return_constraint: 0.96931 w_k1_ADM_s01 - w_k1_ADM_s013
                                 + 0.99 b_k1_ADM_s013 - 1.01 s_k1_ADM_s013  = 0
 w_k1_ADM_s014_return_constraint: 1.0641 w_k1_ADM_s01 - w_k1_ADM_s014
                                 + 0.99 b_k1_ADM_s014 - 1.01 s_k1_ADM_s014  = 0
 w_k1_ADM_s015_return_constraint: 1.44522 w_k1_ADM_s01 - w_k1_ADM_s015
                                 + 0.99 b_k1_ADM_s015 - 1.01 s_k1_ADM_s015  = 0
 w_k1_ADM_s016_return_constraint: 1.23057 w_k1_ADM_s01 - w_k1_ADM_s016
                                 + 0.99 b_k1_ADM_s016 - 1.01 s_k1_ADM_s016  = 0
 w_k1_ADM_s017_return_constraint: 0.788296 w_k1_ADM_s01 - w_k1_ADM_s017
                                 + 0.99 b_k1_ADM_s017 - 1.01 s_k1_ADM_s017  = 0
 w_k1_ADM_s100_return_constraint: 0.980172 w_k1_ADM_s10 - w_k1_ADM_s100
                                 + 0.99 b_k1_ADM_s100 - 1.01 s_k1_ADM_s100  = 0
 w_k1_ADM_s101_return_constraint: 0.883709 w_k1_ADM_s10 - w_k1_ADM_s101
                                 + 0.99 b_k1_ADM_s101 - 1.01 s_k1_ADM_s101  = 0
 w_k1_ADM_s102_return_constraint: 0.827959 w_k1_ADM_s10 - w_k1_ADM_s102
                                 + 0.99 b_k1_ADM_s102 - 1.01 s_k1_ADM_s102  = 0
 w_k1_ADM_s103_return_constraint: 0.651152 w_k1_ADM_s10 - w_k1_ADM_s103
                                 + 0.99 b_k1_ADM_s103 - 1.01 s_k1_ADM_s103  = 0
 w_k1_ADM_s104_return_constraint: 1.01496 w_k1_ADM_s10 - w_k1_ADM_s104
                                 + 0.99 b_k1_ADM_s104 - 1.01 s_k1_ADM_s104  = 0
 w_k1_ADM_s105_return_constraint: 1.04475 w_k1_ADM_s10 - w_k1_ADM_s105
                                 + 0.99 b_k1_ADM_s105 - 1.01 s_k1_ADM_s105  = 0
 w_k1_ADM_s106_return_constraint: 1.43943 w_k1_ADM_s10 - w_k1_ADM_s106
                                 + 0.99 b_k1_ADM_s106 - 1.01 s_k1_ADM_s106  = 0
 w_k1_ADM_s107_return_constraint: 0.879606 w_k1_ADM_s10 - w_k1_ADM_s107
                                 + 0.99 b_k1_ADM_s107 - 1.01 s_k1_ADM_s107  = 0
 w_k1_ADM_s110_return_constraint: 0.717586 w_k1_ADM_s11 - w_k1_ADM_s110
                                 + 0.99 b_k1_ADM_s110 - 1.01 s_k1_ADM_s110  = 0
 w_k1_ADM_s111_return_constraint: 0.784226 w_k1_ADM_s11 - w_k1_ADM_s111
                                 + 0.99 b_k1_ADM_s111 - 1.01 s_k1_ADM_s111  = 0
 w_k1_ADM_s112_return_constraint: 1.3597 w_k1_ADM_s11 - w_k1_ADM_s112
                                 + 0.99 b_k1_ADM_s112 - 1.01 s_k1_ADM_s112  = 0
 w_k1_ADM_s113_return_constraint: 1.01739 w_k1_ADM_s11 - w_k1_ADM_s113
                                 + 0.99 b_k1_ADM_s113 - 1.01 s_k1_ADM_s113  = 0
 w_k1_ADM_s114_return_constraint: 0.624765 w_k1_ADM_s11 - w_k1_ADM_s114
                                 + 0.99 b_k1_ADM_s114 - 1.01 s_k1_ADM_s114  = 0
 w_k1_ADM_s115_return_constraint: 1.0964 w_k1_ADM_s11 - w_k1_ADM_s115
                                 + 0.99 b_k1_ADM_s115 - 1.01 s_k1_ADM_s115  = 0
 w_k1_ADM_s116_return_constraint: 0.703069 w_k1_ADM_s11 - w_k1_ADM_s116
                                 + 0.99 b_k1_ADM_s116 - 1.01 s_k1_ADM_s116  = 0
 w_k1_ADM_s117_return_constraint: 1.0719 w_k1_ADM_s11 - w_k1_ADM_s117
                                 + 0.99 b_k1_ADM_s117 - 1.01 s_k1_ADM_s117  = 0
 w_k1_AXP_s0_return_constraint:  1.1598 w_0_AXP - w_k1_AXP_s0
                                 + 0.99 b_k1_AXP_s0 - 1.01 s_k1_AXP_s0  = 0
 w_k1_AXP_s1_return_constraint:  0.844261 w_0_AXP - w_k1_AXP_s1
                                 + 0.99 b_k1_AXP_s1 - 1.01 s_k1_AXP_s1  = 0
 w_k1_AXP_s00_return_constraint: 1.3011 w_k1_AXP_s0 - w_k1_AXP_s00
                                 + 0.99 b_k1_AXP_s00 - 1.01 s_k1_AXP_s00  = 0
 w_k1_AXP_s01_return_constraint: 0.837366 w_k1_AXP_s0 - w_k1_AXP_s01
                                 + 0.99 b_k1_AXP_s01 - 1.01 s_k1_AXP_s01  = 0
 w_k1_AXP_s10_return_constraint: 0.827934 w_k1_AXP_s1 - w_k1_AXP_s10
                                 + 0.99 b_k1_AXP_s10 - 1.01 s_k1_AXP_s10  = 0
 w_k1_AXP_s11_return_constraint: 1.11238 w_k1_AXP_s1 - w_k1_AXP_s11
                                 + 0.99 b_k1_AXP_s11 - 1.01 s_k1_AXP_s11  = 0
 w_k1_AXP_s000_return_constraint: 1.36415 w_k1_AXP_s00 - w_k1_AXP_s000
                                 + 0.99 b_k1_AXP_s000 - 1.01 s_k1_AXP_s000  = 0
 w_k1_AXP_s001_return_constraint: 0.784676 w_k1_AXP_s00 - w_k1_AXP_s001
                                 + 0.99 b_k1_AXP_s001 - 1.01 s_k1_AXP_s001  = 0
 w_k1_AXP_s002_return_constraint: 0.951977 w_k1_AXP_s00 - w_k1_AXP_s002
                                 + 0.99 b_k1_AXP_s002 - 1.01 s_k1_AXP_s002  = 0
 w_k1_AXP_s003_return_constraint: 0.726777 w_k1_AXP_s00 - w_k1_AXP_s003
                                 + 0.99 b_k1_AXP_s003 - 1.01 s_k1_AXP_s003  = 0
 w_k1_AXP_s004_return_constraint: 0.912732 w_k1_AXP_s00 - w_k1_AXP_s004
                                 + 0.99 b_k1_AXP_s004 - 1.01 s_k1_AXP_s004  = 0
 w_k1_AXP_s005_return_constraint: 0.889354 w_k1_AXP_s00 - w_k1_AXP_s005
                                 + 0.99 b_k1_AXP_s005 - 1.01 s_k1_AXP_s005  = 0
 w_k1_AXP_s006_return_constraint: 0.88631 w_k1_AXP_s00 - w_k1_AXP_s006
                                 + 0.99 b_k1_AXP_s006 - 1.01 s_k1_AXP_s006  = 0
 w_k1_AXP_s007_return_constraint: 0.89389 w_k1_AXP_s00 - w_k1_AXP_s007
                                 + 0.99 b_k1_AXP_s007 - 1.01 s_k1_AXP_s007  = 0
 w_k1_AXP_s010_return_constraint: 1.00137 w_k1_AXP_s01 - w_k1_AXP_s010
                                 + 0.99 b_k1_AXP_s010 - 1.01 s_k1_AXP_s010  = 0
 w_k1_AXP_s011_return_constraint: 0.633123 w_k1_AXP_s01 - w_k1_AXP_s011
                                 + 0.99 b_k1_AXP_s011 - 1.01 s_k1_AXP_s011  = 0
 w_k1_AXP_s012_return_constraint: 0.965569 w_k1_AXP_s01 - w_k1_AXP_s012
                                 + 0.99 b_k1_AXP_s012 - 1.01 s_k1_AXP_s012  = 0
 w_k1_AXP_s013_return_constraint: 0.753149 w_k1_AXP_s01 - w_k1_AXP_s013
                                 + 0.99 b_k1_AXP_s013 - 1.01 s_k1_AXP_s013  = 0
 w_k1_AXP_s014_return_constraint: 1.22264 w_k1_AXP_s01 - w_k1_AXP_s014
                                 + 0.99 b_k1_AXP_s014 - 1.01 s_k1_AXP_s014  = 0
 w_k1_AXP_s015_return_constraint: 1.8383 w_k1_AXP_s01 - w_k1_AXP_s015
                                 + 0.99 b_k1_AXP_s015 - 1.01 s_k1_AXP_s015  = 0
 w_k1_AXP_s016_return_constraint: 1.52565 w_k1_AXP_s01 - w_k1_AXP_s016
                                 + 0.99 b_k1_AXP_s016 - 1.01 s_k1_AXP_s016  = 0
 w_k1_AXP_s017_return_constraint: 0.895987 w_k1_AXP_s01 - w_k1_AXP_s017
                                 + 0.99 b_k1_AXP_s017 - 1.01 s_k1_AXP_s017  = 0
 w_k1_AXP_s100_return_constraint: 0.963487 w_k1_AXP_s10 - w_k1_AXP_s100
                                 + 0.99 b_k1_AXP_s100 - 1.01 s_k1_AXP_s100  = 0
 w_k1_AXP_s101_return_constraint: 1.02647 w_k1_AXP_s10 - w_k1_AXP_s101
                                 + 0.99 b_k1_AXP_s101 - 1.01 s_k1_AXP_s101  = 0
 w_k1_AXP_s102_return_constraint: 0.858819 w_k1_AXP_s10 - w_k1_AXP_s102
                                 + 0.99 b_k1_AXP_s102 - 1.01 s_k1_AXP_s102  = 0
 w_k1_AXP_s103_return_constraint: 0.595685 w_k1_AXP_s10 - w_k1_AXP_s103
                                 + 0.99 b_k1_AXP_s103 - 1.01 s_k1_AXP_s103  = 0
 w_k1_AXP_s104_return_constraint: 1.22149 w_k1_AXP_s10 - w_k1_AXP_s104
                                 + 0.99 b_k1_AXP_s104 - 1.01 s_k1_AXP_s104  = 0
 w_k1_AXP_s105_return_constraint: 1.25608 w_k1_AXP_s10 - w_k1_AXP_s105
                                 + 0.99 b_k1_AXP_s105 - 1.01 s_k1_AXP_s105  = 0
 w_k1_AXP_s106_return_constraint: 1.71458 w_k1_AXP_s10 - w_k1_AXP_s106
                                 + 0.99 b_k1_AXP_s106 - 1.01 s_k1_AXP_s106  = 0
 w_k1_AXP_s107_return_constraint: 0.993235 w_k1_AXP_s10 - w_k1_AXP_s107
                                 + 0.99 b_k1_AXP_s107 - 1.01 s_k1_AXP_s107  = 0
 w_k1_AXP_s110_return_constraint: 0.767541 w_k1_AXP_s11 - w_k1_AXP_s110
                                 + 0.99 b_k1_AXP_s110 - 1.01 s_k1_AXP_s110  = 0
 w_k1_AXP_s111_return_constraint: 0.750119 w_k1_AXP_s11 - w_k1_AXP_s111
                                 + 0.99 b_k1_AXP_s111 - 1.01 s_k1_AXP_s111  = 0
 w_k1_AXP_s112_return_constraint: 1.49393 w_k1_AXP_s11 - w_k1_AXP_s112
                                 + 0.99 b_k1_AXP_s112 - 1.01 s_k1_AXP_s112  = 0
 w_k1_AXP_s113_return_constraint: 1.03525 w_k1_AXP_s11 - w_k1_AXP_s113
                                 + 0.99 b_k1_AXP_s113 - 1.01 s_k1_AXP_s113  = 0
 w_k1_AXP_s114_return_constraint: 0.559106 w_k1_AXP_s11 - w_k1_AXP_s114
                                 + 0.99 b_k1_AXP_s114 - 1.01 s_k1_AXP_s114  = 0
 w_k1_AXP_s115_return_constraint: 1.14969 w_k1_AXP_s11 - w_k1_AXP_s115
                                 + 0.99 b_k1_AXP_s115 - 1.01 s_k1_AXP_s115  = 0
 w_k1_AXP_s116_return_constraint: 0.76091 w_k1_AXP_s11 - w_k1_AXP_s116
                                 + 0.99 b_k1_AXP_s116 - 1.01 s_k1_AXP_s116  = 0
 w_k1_AXP_s117_return_constraint: 1.30412 w_k1_AXP_s11 - w_k1_AXP_s117
                                 + 0.99 b_k1_AXP_s117 - 1.01 s_k1_AXP_s117  = 0
 w_k1_DIS_s0_return_constraint:  1.19651 w_0_DIS - w_k1_DIS_s0
                                 + 0.99 b_k1_DIS_s0 - 1.01 s_k1_DIS_s0  = 0
 w_k1_DIS_s1_return_constraint:  0.817514 w_0_DIS - w_k1_DIS_s1
                                 + 0.99 b_k1_DIS_s1 - 1.01 s_k1_DIS_s1  = 0
 w_k1_DIS_s00_return_constraint: 1.4556 w_k1_DIS_s0 - w_k1_DIS_s00
                                 + 0.99 b_k1_DIS_s00 - 1.01 s_k1_DIS_s00  = 0
 w_k1_DIS_s01_return_constraint: 0.872927 w_k1_DIS_s0 - w_k1_DIS_s01
                                 + 0.99 b_k1_DIS_s01 - 1.01 s_k1_DIS_s01  = 0
 w_k1_DIS_s10_return_constraint: 0.837693 w_k1_DIS_s1 - w_k1_DIS_s10
                                 + 0.99 b_k1_DIS_s10 - 1.01 s_k1_DIS_s10  = 0
 w_k1_DIS_s11_return_constraint: 1.09877 w_k1_DIS_s1 - w_k1_DIS_s11
                                 + 0.99 b_k1_DIS_s11 - 1.01 s_k1_DIS_s11  = 0
 w_k1_DIS_s000_return_constraint: 1.40594 w_k1_DIS_s00 - w_k1_DIS_s000
                                 + 0.99 b_k1_DIS_s000 - 1.01 s_k1_DIS_s000  = 0
 w_k1_DIS_s001_return_constraint: 0.682314 w_k1_DIS_s00 - w_k1_DIS_s001
                                 + 0.99 b_k1_DIS_s001 - 1.01 s_k1_DIS_s001  = 0
 w_k1_DIS_s002_return_constraint: 0.732766 w_k1_DIS_s00 - w_k1_DIS_s002
                                 + 0.99 b_k1_DIS_s002 - 1.01 s_k1_DIS_s002  = 0
 w_k1_DIS_s003_return_constraint: 0.582931 w_k1_DIS_s00 - w_k1_DIS_s003
                                 + 0.99 b_k1_DIS_s003 - 1.01 s_k1_DIS_s003  = 0
 w_k1_DIS_s004_return_constraint: 1.04435 w_k1_DIS_s00 - w_k1_DIS_s004
                                 + 0.99 b_k1_DIS_s004 - 1.01 s_k1_DIS_s004  = 0
 w_k1_DIS_s005_return_constraint: 0.80669 w_k1_DIS_s00 - w_k1_DIS_s005
                                 + 0.99 b_k1_DIS_s005 - 1.01 s_k1_DIS_s005  = 0
 w_k1_DIS_s006_return_constraint: 0.794014 w_k1_DIS_s00 - w_k1_DIS_s006
                                 + 0.99 b_k1_DIS_s006 - 1.01 s_k1_DIS_s006  = 0
 w_k1_DIS_s007_return_constraint: 1.06236 w_k1_DIS_s00 - w_k1_DIS_s007
                                 + 0.99 b_k1_DIS_s007 - 1.01 s_k1_DIS_s007  = 0
 w_k1_DIS_s010_return_constraint: 1.0356 w_k1_DIS_s01 - w_k1_DIS_s010
                                 + 0.99 b_k1_DIS_s010 - 1.01 s_k1_DIS_s010  = 0
 w_k1_DIS_s011_return_constraint: 0.566579 w_k1_DIS_s01 - w_k1_DIS_s011
                                 + 0.99 b_k1_DIS_s011 - 1.01 s_k1_DIS_s011  = 0
 w_k1_DIS_s012_return_constraint: 0.922073 w_k1_DIS_s01 - w_k1_DIS_s012
                                 + 0.99 b_k1_DIS_s012 - 1.01 s_k1_DIS_s012  = 0
 w_k1_DIS_s013_return_constraint: 0.798975 w_k1_DIS_s01 - w_k1_DIS_s013
                                 + 0.99 b_k1_DIS_s013 - 1.01 s_k1_DIS_s013  = 0
 w_k1_DIS_s014_return_constraint: 1.29337 w_k1_DIS_s01 - w_k1_DIS_s014
                                 + 0.99 b_k1_DIS_s014 - 1.01 s_k1_DIS_s014  = 0
 w_k1_DIS_s015_return_constraint: 1.78375 w_k1_DIS_s01 - w_k1_DIS_s015
                                 + 0.99 b_k1_DIS_s015 - 1.01 s_k1_DIS_s015  = 0
 w_k1_DIS_s016_return_constraint: 0.999837 w_k1_DIS_s01 - w_k1_DIS_s016
                                 + 0.99 b_k1_DIS_s016 - 1.01 s_k1_DIS_s016  = 0
 w_k1_DIS_s017_return_constraint: 0.662153 w_k1_DIS_s01 - w_k1_DIS_s017
                                 + 0.99 b_k1_DIS_s017 - 1.01 s_k1_DIS_s017  = 0
 w_k1_DIS_s100_return_constraint: 0.965004 w_k1_DIS_s10 - w_k1_DIS_s100
                                 + 0.99 b_k1_DIS_s100 - 1.01 s_k1_DIS_s100  = 0
 w_k1_DIS_s101_return_constraint: 1.09026 w_k1_DIS_s10 - w_k1_DIS_s101
                                 + 0.99 b_k1_DIS_s101 - 1.01 s_k1_DIS_s101  = 0
 w_k1_DIS_s102_return_constraint: 0.849231 w_k1_DIS_s10 - w_k1_DIS_s102
                                 + 0.99 b_k1_DIS_s102 - 1.01 s_k1_DIS_s102  = 0
 w_k1_DIS_s103_return_constraint: 0.597939 w_k1_DIS_s10 - w_k1_DIS_s103
                                 + 0.99 b_k1_DIS_s103 - 1.01 s_k1_DIS_s103  = 0
 w_k1_DIS_s104_return_constraint: 1.20895 w_k1_DIS_s10 - w_k1_DIS_s104
                                 + 0.99 b_k1_DIS_s104 - 1.01 s_k1_DIS_s104  = 0
 w_k1_DIS_s105_return_constraint: 1.0474 w_k1_DIS_s10 - w_k1_DIS_s105
                                 + 0.99 b_k1_DIS_s105 - 1.01 s_k1_DIS_s105  = 0
 w_k1_DIS_s106_return_constraint: 1.90991 w_k1_DIS_s10 - w_k1_DIS_s106
                                 + 0.99 b_k1_DIS_s106 - 1.01 s_k1_DIS_s106  = 0
 w_k1_DIS_s107_return_constraint: 1.31345 w_k1_DIS_s10 - w_k1_DIS_s107
                                 + 0.99 b_k1_DIS_s107 - 1.01 s_k1_DIS_s107  = 0
 w_k1_DIS_s110_return_constraint: 0.707616 w_k1_DIS_s11 - w_k1_DIS_s110
                                 + 0.99 b_k1_DIS_s110 - 1.01 s_k1_DIS_s110  = 0
 w_k1_DIS_s111_return_constraint: 0.718243 w_k1_DIS_s11 - w_k1_DIS_s111
                                 + 0.99 b_k1_DIS_s111 - 1.01 s_k1_DIS_s111  = 0
 w_k1_DIS_s112_return_constraint: 1.85199 w_k1_DIS_s11 - w_k1_DIS_s112
                                 + 0.99 b_k1_DIS_s112 - 1.01 s_k1_DIS_s112  = 0
 w_k1_DIS_s113_return_constraint: 1.10655 w_k1_DIS_s11 - w_k1_DIS_s113
                                 + 0.99 b_k1_DIS_s113 - 1.01 s_k1_DIS_s113  = 0
 w_k1_DIS_s114_return_constraint: 0.526351 w_k1_DIS_s11 - w_k1_DIS_s114
                                 + 0.99 b_k1_DIS_s114 - 1.01 s_k1_DIS_s114  = 0
 w_k1_DIS_s115_return_constraint: 1.11534 w_k1_DIS_s11 - w_k1_DIS_s115
                                 + 0.99 b_k1_DIS_s115 - 1.01 s_k1_DIS_s115  = 0
 w_k1_DIS_s116_return_constraint: 0.771445 w_k1_DIS_s11 - w_k1_DIS_s116
                                 + 0.99 b_k1_DIS_s116 - 1.01 s_k1_DIS_s116  = 0
 w_k1_DIS_s117_return_constraint: 1.52315 w_k1_DIS_s11 - w_k1_DIS_s117
                                 + 0.99 b_k1_DIS_s117 - 1.01 s_k1_DIS_s117  = 0
 w_k1_FDX_s0_return_constraint:  1.12032 w_0_FDX - w_k1_FDX_s0
                                 + 0.99 b_k1_FDX_s0 - 1.01 s_k1_FDX_s0  = 0
 w_k1_FDX_s1_return_constraint:  0.889756 w_0_FDX - w_k1_FDX_s1
                                 + 0.99 b_k1_FDX_s1 - 1.01 s_k1_FDX_s1  = 0
 w_k1_FDX_s00_return_constraint: 1.1995 w_k1_FDX_s0 - w_k1_FDX_s00
                                 + 0.99 b_k1_FDX_s00 - 1.01 s_k1_FDX_s00  = 0
 w_k1_FDX_s01_return_constraint: 0.923147 w_k1_FDX_s0 - w_k1_FDX_s01
                                 + 0.99 b_k1_FDX_s01 - 1.01 s_k1_FDX_s01  = 0
 w_k1_FDX_s10_return_constraint: 0.940836 w_k1_FDX_s1 - w_k1_FDX_s10
                                 + 0.99 b_k1_FDX_s10 - 1.01 s_k1_FDX_s10  = 0
 w_k1_FDX_s11_return_constraint: 1.0563 w_k1_FDX_s1 - w_k1_FDX_s11
                                 + 0.99 b_k1_FDX_s11 - 1.01 s_k1_FDX_s11  = 0
 w_k1_FDX_s000_return_constraint: 1.32474 w_k1_FDX_s00 - w_k1_FDX_s000
                                 + 0.99 b_k1_FDX_s000 - 1.01 s_k1_FDX_s000  = 0
 w_k1_FDX_s001_return_constraint: 0.811912 w_k1_FDX_s00 - w_k1_FDX_s001
                                 + 0.99 b_k1_FDX_s001 - 1.01 s_k1_FDX_s001  = 0
 w_k1_FDX_s002_return_constraint: 0.856746 w_k1_FDX_s00 - w_k1_FDX_s002
                                 + 0.99 b_k1_FDX_s002 - 1.01 s_k1_FDX_s002  = 0
 w_k1_FDX_s003_return_constraint: 0.64647 w_k1_FDX_s00 - w_k1_FDX_s003
                                 + 0.99 b_k1_FDX_s003 - 1.01 s_k1_FDX_s003  = 0
 w_k1_FDX_s004_return_constraint: 1.11071 w_k1_FDX_s00 - w_k1_FDX_s004
                                 + 0.99 b_k1_FDX_s004 - 1.01 s_k1_FDX_s004  = 0
 w_k1_FDX_s005_return_constraint: 1.07838 w_k1_FDX_s00 - w_k1_FDX_s005
                                 + 0.99 b_k1_FDX_s005 - 1.01 s_k1_FDX_s005  = 0
 w_k1_FDX_s006_return_constraint: 0.985602 w_k1_FDX_s00 - w_k1_FDX_s006
                                 + 0.99 b_k1_FDX_s006 - 1.01 s_k1_FDX_s006  = 0
 w_k1_FDX_s007_return_constraint: 0.985048 w_k1_FDX_s00 - w_k1_FDX_s007
                                 + 0.99 b_k1_FDX_s007 - 1.01 s_k1_FDX_s007  = 0
 w_k1_FDX_s010_return_constraint: 1.04553 w_k1_FDX_s01 - w_k1_FDX_s010
                                 + 0.99 b_k1_FDX_s010 - 1.01 s_k1_FDX_s010  = 0
 w_k1_FDX_s011_return_constraint: 0.666368 w_k1_FDX_s01 - w_k1_FDX_s011
                                 + 0.99 b_k1_FDX_s011 - 1.01 s_k1_FDX_s011  = 0
 w_k1_FDX_s012_return_constraint: 0.933417 w_k1_FDX_s01 - w_k1_FDX_s012
                                 + 0.99 b_k1_FDX_s012 - 1.01 s_k1_FDX_s012  = 0
 w_k1_FDX_s013_return_constraint: 0.92887 w_k1_FDX_s01 - w_k1_FDX_s013
                                 + 0.99 b_k1_FDX_s013 - 1.01 s_k1_FDX_s013  = 0
 w_k1_FDX_s014_return_constraint: 1.07422 w_k1_FDX_s01 - w_k1_FDX_s014
                                 + 0.99 b_k1_FDX_s014 - 1.01 s_k1_FDX_s014  = 0
 w_k1_FDX_s015_return_constraint: 1.47864 w_k1_FDX_s01 - w_k1_FDX_s015
                                 + 0.99 b_k1_FDX_s015 - 1.01 s_k1_FDX_s015  = 0
 w_k1_FDX_s016_return_constraint: 1.10462 w_k1_FDX_s01 - w_k1_FDX_s016
                                 + 0.99 b_k1_FDX_s016 - 1.01 s_k1_FDX_s016  = 0
 w_k1_FDX_s017_return_constraint: 0.731767 w_k1_FDX_s01 - w_k1_FDX_s017
                                 + 0.99 b_k1_FDX_s017 - 1.01 s_k1_FDX_s017  = 0
 w_k1_FDX_s100_return_constraint: 1.09858 w_k1_FDX_s10 - w_k1_FDX_s100
                                 + 0.99 b_k1_FDX_s100 - 1.01 s_k1_FDX_s100  = 0
 w_k1_FDX_s101_return_constraint: 1.00912 w_k1_FDX_s10 - w_k1_FDX_s101
                                 + 0.99 b_k1_FDX_s101 - 1.01 s_k1_FDX_s101  = 0
 w_k1_FDX_s102_return_constraint: 0.877637 w_k1_FDX_s10 - w_k1_FDX_s102
                                 + 0.99 b_k1_FDX_s102 - 1.01 s_k1_FDX_s102  = 0
 w_k1_FDX_s103_return_constraint: 0.662349 w_k1_FDX_s10 - w_k1_FDX_s103
                                 + 0.99 b_k1_FDX_s103 - 1.01 s_k1_FDX_s103  = 0
 w_k1_FDX_s104_return_constraint: 1.17386 w_k1_FDX_s10 - w_k1_FDX_s104
                                 + 0.99 b_k1_FDX_s104 - 1.01 s_k1_FDX_s104  = 0
 w_k1_FDX_s105_return_constraint: 0.967819 w_k1_FDX_s10 - w_k1_FDX_s105
                                 + 0.99 b_k1_FDX_s105 - 1.01 s_k1_FDX_s105  = 0
 w_k1_FDX_s106_return_constraint: 1.60033 w_k1_FDX_s10 - w_k1_FDX_s106
                                 + 0.99 b_k1_FDX_s106 - 1.01 s_k1_FDX_s106  = 0
 w_k1_FDX_s107_return_constraint: 0.966327 w_k1_FDX_s10 - w_k1_FDX_s107
                                 + 0.99 b_k1_FDX_s107 - 1.01 s_k1_FDX_s107  = 0
 w_k1_FDX_s110_return_constraint: 0.82019 w_k1_FDX_s11 - w_k1_FDX_s110
                                 + 0.99 b_k1_FDX_s110 - 1.01 s_k1_FDX_s110  = 0
 w_k1_FDX_s111_return_constraint: 0.724327 w_k1_FDX_s11 - w_k1_FDX_s111
                                 + 0.99 b_k1_FDX_s111 - 1.01 s_k1_FDX_s111  = 0
 w_k1_FDX_s112_return_constraint: 1.53615 w_k1_FDX_s11 - w_k1_FDX_s112
                                 + 0.99 b_k1_FDX_s112 - 1.01 s_k1_FDX_s112  = 0
 w_k1_FDX_s113_return_constraint: 0.946545 w_k1_FDX_s11 - w_k1_FDX_s113
                                 + 0.99 b_k1_FDX_s113 - 1.01 s_k1_FDX_s113  = 0
 w_k1_FDX_s114_return_constraint: 0.620053 w_k1_FDX_s11 - w_k1_FDX_s114
                                 + 0.99 b_k1_FDX_s114 - 1.01 s_k1_FDX_s114  = 0
 w_k1_FDX_s115_return_constraint: 1.15578 w_k1_FDX_s11 - w_k1_FDX_s115
                                 + 0.99 b_k1_FDX_s115 - 1.01 s_k1_FDX_s115  = 0
 w_k1_FDX_s116_return_constraint: 0.798754 w_k1_FDX_s11 - w_k1_FDX_s116
                                 + 0.99 b_k1_FDX_s116 - 1.01 s_k1_FDX_s116  = 0
 w_k1_FDX_s117_return_constraint: 1.30658 w_k1_FDX_s11 - w_k1_FDX_s117
                                 + 0.99 b_k1_FDX_s117 - 1.01 s_k1_FDX_s117  = 0
 w_k1_IBM_s0_return_constraint:  1.14909 w_0_IBM - w_k1_IBM_s0
                                 + 0.99 b_k1_IBM_s0 - 1.01 s_k1_IBM_s0  = 0
 w_k1_IBM_s1_return_constraint:  0.859183 w_0_IBM - w_k1_IBM_s1
                                 + 0.99 b_k1_IBM_s1 - 1.01 s_k1_IBM_s1  = 0
 w_k1_IBM_s00_return_constraint: 1.3112 w_k1_IBM_s0 - w_k1_IBM_s00
                                 + 0.99 b_k1_IBM_s00 - 1.01 s_k1_IBM_s00  = 0
 w_k1_IBM_s01_return_constraint: 0.910407 w_k1_IBM_s0 - w_k1_IBM_s01
                                 + 0.99 b_k1_IBM_s01 - 1.01 s_k1_IBM_s01  = 0
 w_k1_IBM_s10_return_constraint: 0.729201 w_k1_IBM_s1 - w_k1_IBM_s10
                                 + 0.99 b_k1_IBM_s10 - 1.01 s_k1_IBM_s10  = 0
 w_k1_IBM_s11_return_constraint: 1.18449 w_k1_IBM_s1 - w_k1_IBM_s11
                                 + 0.99 b_k1_IBM_s11 - 1.01 s_k1_IBM_s11  = 0
 w_k1_IBM_s000_return_constraint: 1.05061 w_k1_IBM_s00 - w_k1_IBM_s000
                                 + 0.99 b_k1_IBM_s000 - 1.01 s_k1_IBM_s000  = 0
 w_k1_IBM_s001_return_constraint: 0.676227 w_k1_IBM_s00 - w_k1_IBM_s001
                                 + 0.99 b_k1_IBM_s001 - 1.01 s_k1_IBM_s001  = 0
 w_k1_IBM_s002_return_constraint: 1.14159 w_k1_IBM_s00 - w_k1_IBM_s002
                                 + 0.99 b_k1_IBM_s002 - 1.01 s_k1_IBM_s002  = 0
 w_k1_IBM_s003_return_constraint: 1.14229 w_k1_IBM_s00 - w_k1_IBM_s003
                                 + 0.99 b_k1_IBM_s003 - 1.01 s_k1_IBM_s003  = 0
 w_k1_IBM_s004_return_constraint: 0.83356 w_k1_IBM_s00 - w_k1_IBM_s004
                                 + 0.99 b_k1_IBM_s004 - 1.01 s_k1_IBM_s004  = 0
 w_k1_IBM_s005_return_constraint: 0.586499 w_k1_IBM_s00 - w_k1_IBM_s005
                                 + 0.99 b_k1_IBM_s005 - 1.01 s_k1_IBM_s005  = 0
 w_k1_IBM_s006_return_constraint: 0.812646 w_k1_IBM_s00 - w_k1_IBM_s006
                                 + 0.99 b_k1_IBM_s006 - 1.01 s_k1_IBM_s006  = 0
 w_k1_IBM_s007_return_constraint: 1.29399 w_k1_IBM_s00 - w_k1_IBM_s007
                                 + 0.99 b_k1_IBM_s007 - 1.01 s_k1_IBM_s007  = 0
 w_k1_IBM_s010_return_constraint: 0.677628 w_k1_IBM_s01 - w_k1_IBM_s010
                                 + 0.99 b_k1_IBM_s010 - 1.01 s_k1_IBM_s010  = 0
 w_k1_IBM_s011_return_constraint: 0.711113 w_k1_IBM_s01 - w_k1_IBM_s011
                                 + 0.99 b_k1_IBM_s011 - 1.01 s_k1_IBM_s011  = 0
 w_k1_IBM_s012_return_constraint: 0.984889 w_k1_IBM_s01 - w_k1_IBM_s012
                                 + 0.99 b_k1_IBM_s012 - 1.01 s_k1_IBM_s012  = 0
 w_k1_IBM_s013_return_constraint: 0.517575 w_k1_IBM_s01 - w_k1_IBM_s013
                                 + 0.99 b_k1_IBM_s013 - 1.01 s_k1_IBM_s013  = 0
 w_k1_IBM_s014_return_constraint: 1.2368 w_k1_IBM_s01 - w_k1_IBM_s014
                                 + 0.99 b_k1_IBM_s014 - 1.01 s_k1_IBM_s014  = 0
 w_k1_IBM_s015_return_constraint: 1.39579 w_k1_IBM_s01 - w_k1_IBM_s015
                                 + 0.99 b_k1_IBM_s015 - 1.01 s_k1_IBM_s015  = 0
 w_k1_IBM_s016_return_constraint: 0.963964 w_k1_IBM_s01 - w_k1_IBM_s016
                                 + 0.99 b_k1_IBM_s016 - 1.01 s_k1_IBM_s016  = 0
 w_k1_IBM_s017_return_constraint: 1.05607 w_k1_IBM_s01 - w_k1_IBM_s017
                                 + 0.99 b_k1_IBM_s017 - 1.01 s_k1_IBM_s017  = 0
 w_k1_IBM_s100_return_constraint: 0.811292 w_k1_IBM_s10 - w_k1_IBM_s100
                                 + 0.99 b_k1_IBM_s100 - 1.01 s_k1_IBM_s100  = 0
 w_k1_IBM_s101_return_constraint: 1.23412 w_k1_IBM_s10 - w_k1_IBM_s101
                                 + 0.99 b_k1_IBM_s101 - 1.01 s_k1_IBM_s101  = 0
 w_k1_IBM_s102_return_constraint: 0.795554 w_k1_IBM_s10 - w_k1_IBM_s102
                                 + 0.99 b_k1_IBM_s102 - 1.01 s_k1_IBM_s102  = 0
 w_k1_IBM_s103_return_constraint: 0.840219 w_k1_IBM_s10 - w_k1_IBM_s103
                                 + 0.99 b_k1_IBM_s103 - 1.01 s_k1_IBM_s103  = 0
 w_k1_IBM_s104_return_constraint: 1.05255 w_k1_IBM_s10 - w_k1_IBM_s104
                                 + 0.99 b_k1_IBM_s104 - 1.01 s_k1_IBM_s104  = 0
 w_k1_IBM_s105_return_constraint: 1.59464 w_k1_IBM_s10 - w_k1_IBM_s105
                                 + 0.99 b_k1_IBM_s105 - 1.01 s_k1_IBM_s105  = 0
 w_k1_IBM_s106_return_constraint: 1.16633 w_k1_IBM_s10 - w_k1_IBM_s106
                                 + 0.99 b_k1_IBM_s106 - 1.01 s_k1_IBM_s106  = 0
 w_k1_IBM_s107_return_constraint: 1.28935 w_k1_IBM_s10 - w_k1_IBM_s107
                                 + 0.99 b_k1_IBM_s107 - 1.01 s_k1_IBM_s107  = 0
 w_k1_IBM_s110_return_constraint: 0.585673 w_k1_IBM_s11 - w_k1_IBM_s110
                                 + 0.99 b_k1_IBM_s110 - 1.01 s_k1_IBM_s110  = 0
 w_k1_IBM_s111_return_constraint: 1.38762 w_k1_IBM_s11 - w_k1_IBM_s111
                                 + 0.99 b_k1_IBM_s111 - 1.01 s_k1_IBM_s111  = 0
 w_k1_IBM_s112_return_constraint: 0.914285 w_k1_IBM_s11 - w_k1_IBM_s112
                                 + 0.99 b_k1_IBM_s112 - 1.01 s_k1_IBM_s112  = 0
 w_k1_IBM_s113_return_constraint: 1.12074 w_k1_IBM_s11 - w_k1_IBM_s113
                                 + 0.99 b_k1_IBM_s113 - 1.01 s_k1_IBM_s113  = 0
 w_k1_IBM_s114_return_constraint: 0.819278 w_k1_IBM_s11 - w_k1_IBM_s114
                                 + 0.99 b_k1_IBM_s114 - 1.01 s_k1_IBM_s114  = 0
 w_k1_IBM_s115_return_constraint: 0.731381 w_k1_IBM_s11 - w_k1_IBM_s115
                                 + 0.99 b_k1_IBM_s115 - 1.01 s_k1_IBM_s115  = 0
 w_k1_IBM_s116_return_constraint: 0.851225 w_k1_IBM_s11 - w_k1_IBM_s116
                                 + 0.99 b_k1_IBM_s116 - 1.01 s_k1_IBM_s116  = 0
 w_k1_IBM_s117_return_constraint: 1.33889 w_k1_IBM_s11 - w_k1_IBM_s117
                                 + 0.99 b_k1_IBM_s117 - 1.01 s_k1_IBM_s117  = 0
 w_k1_INTC_s0_return_constraint: 1.08275 w_0_INTC - w_k1_INTC_s0
                                 + 0.99 b_k1_INTC_s0 - 1.01 s_k1_INTC_s0  = 0
 w_k1_INTC_s1_return_constraint: 0.921849 w_0_INTC - w_k1_INTC_s1
                                 + 0.99 b_k1_INTC_s1 - 1.01 s_k1_INTC_s1  = 0
 w_k1_INTC_s00_return_constraint: 1.23714 w_k1_INTC_s0 - w_k1_INTC_s00
                                 + 0.99 b_k1_INTC_s00 - 1.01 s_k1_INTC_s00  = 0
 w_k1_INTC_s01_return_constraint: 0.965721 w_k1_INTC_s0 - w_k1_INTC_s01
                                 + 0.99 b_k1_INTC_s01 - 1.01 s_k1_INTC_s01  = 0
 w_k1_INTC_s10_return_constraint: 0.900135 w_k1_INTC_s1 - w_k1_INTC_s10
                                 + 0.99 b_k1_INTC_s10 - 1.01 s_k1_INTC_s10  = 0
 w_k1_INTC_s11_return_constraint: 1.00895 w_k1_INTC_s1 - w_k1_INTC_s11
                                 + 0.99 b_k1_INTC_s11 - 1.01 s_k1_INTC_s11  = 0
 w_k1_INTC_s000_return_constraint: 1.25864 w_k1_INTC_s00 - w_k1_INTC_s000
                                 + 0.99 b_k1_INTC_s000 - 1.01 s_k1_INTC_s000
                                  = 0
 w_k1_INTC_s001_return_constraint: 0.834101 w_k1_INTC_s00 - w_k1_INTC_s001
                                 + 0.99 b_k1_INTC_s001 - 1.01 s_k1_INTC_s001
                                  = 0
 w_k1_INTC_s002_return_constraint: 0.729398 w_k1_INTC_s00 - w_k1_INTC_s002
                                 + 0.99 b_k1_INTC_s002 - 1.01 s_k1_INTC_s002
                                  = 0
 w_k1_INTC_s003_return_constraint: 0.76708 w_k1_INTC_s00 - w_k1_INTC_s003
                                 + 0.99 b_k1_INTC_s003 - 1.01 s_k1_INTC_s003
                                  = 0
 w_k1_INTC_s004_return_constraint: 1.07428 w_k1_INTC_s00 - w_k1_INTC_s004
                                 + 0.99 b_k1_INTC_s004 - 1.01 s_k1_INTC_s004
                                  = 0
 w_k1_INTC_s005_return_constraint: 0.970137 w_k1_INTC_s00 - w_k1_INTC_s005
                                 + 0.99 b_k1_INTC_s005 - 1.01 s_k1_INTC_s005
                                  = 0
 w_k1_INTC_s006_return_constraint: 0.901635 w_k1_INTC_s00 - w_k1_INTC_s006
                                 + 0.99 b_k1_INTC_s006 - 1.01 s_k1_INTC_s006
                                  = 0
 w_k1_INTC_s007_return_constraint: 0.921333 w_k1_INTC_s00 - w_k1_INTC_s007
                                 + 0.99 b_k1_INTC_s007 - 1.01 s_k1_INTC_s007
                                  = 0
 w_k1_INTC_s010_return_constraint: 1.06293 w_k1_INTC_s01 - w_k1_INTC_s010
                                 + 0.99 b_k1_INTC_s010 - 1.01 s_k1_INTC_s010
                                  = 0
 w_k1_INTC_s011_return_constraint: 0.743067 w_k1_INTC_s01 - w_k1_INTC_s011
                                 + 0.99 b_k1_INTC_s011 - 1.01 s_k1_INTC_s011
                                  = 0
 w_k1_INTC_s012_return_constraint: 0.932396 w_k1_INTC_s01 - w_k1_INTC_s012
                                 + 0.99 b_k1_INTC_s012 - 1.01 s_k1_INTC_s012
                                  = 0
 w_k1_INTC_s013_return_constraint: 0.928215 w_k1_INTC_s01 - w_k1_INTC_s013
                                 + 0.99 b_k1_INTC_s013 - 1.01 s_k1_INTC_s013
                                  = 0
 w_k1_INTC_s014_return_constraint: 1.16804 w_k1_INTC_s01 - w_k1_INTC_s014
                                 + 0.99 b_k1_INTC_s014 - 1.01 s_k1_INTC_s014
                                  = 0
 w_k1_INTC_s015_return_constraint: 1.17238 w_k1_INTC_s01 - w_k1_INTC_s015
                                 + 0.99 b_k1_INTC_s015 - 1.01 s_k1_INTC_s015
                                  = 0
 w_k1_INTC_s016_return_constraint: 0.991191 w_k1_INTC_s01 - w_k1_INTC_s016
                                 + 0.99 b_k1_INTC_s016 - 1.01 s_k1_INTC_s016
                                  = 0
 w_k1_INTC_s017_return_constraint: 0.841695 w_k1_INTC_s01 - w_k1_INTC_s017
                                 + 0.99 b_k1_INTC_s017 - 1.01 s_k1_INTC_s017
                                  = 0
 w_k1_INTC_s100_return_constraint: 1.02642 w_k1_INTC_s10 - w_k1_INTC_s100
                                 + 0.99 b_k1_INTC_s100 - 1.01 s_k1_INTC_s100
                                  = 0
 w_k1_INTC_s101_return_constraint: 1.17418 w_k1_INTC_s10 - w_k1_INTC_s101
                                 + 0.99 b_k1_INTC_s101 - 1.01 s_k1_INTC_s101
                                  = 0
 w_k1_INTC_s102_return_constraint: 0.851774 w_k1_INTC_s10 - w_k1_INTC_s102
                                 + 0.99 b_k1_INTC_s102 - 1.01 s_k1_INTC_s102
                                  = 0
 w_k1_INTC_s103_return_constraint: 0.788638 w_k1_INTC_s10 - w_k1_INTC_s103
                                 + 0.99 b_k1_INTC_s103 - 1.01 s_k1_INTC_s103
                                  = 0
 w_k1_INTC_s104_return_constraint: 1.19138 w_k1_INTC_s10 - w_k1_INTC_s104
                                 + 0.99 b_k1_INTC_s104 - 1.01 s_k1_INTC_s104
                                  = 0
 w_k1_INTC_s105_return_constraint: 1.1197 w_k1_INTC_s10 - w_k1_INTC_s105
                                 + 0.99 b_k1_INTC_s105 - 1.01 s_k1_INTC_s105
                                  = 0
 w_k1_INTC_s106_return_constraint: 1.52741 w_k1_INTC_s10 - w_k1_INTC_s106
                                 + 0.99 b_k1_INTC_s106 - 1.01 s_k1_INTC_s106
                                  = 0
 w_k1_INTC_s107_return_constraint: 0.957876 w_k1_INTC_s10 - w_k1_INTC_s107
                                 + 0.99 b_k1_INTC_s107 - 1.01 s_k1_INTC_s107
                                  = 0
 w_k1_INTC_s110_return_constraint: 0.924876 w_k1_INTC_s11 - w_k1_INTC_s110
                                 + 0.99 b_k1_INTC_s110 - 1.01 s_k1_INTC_s110
                                  = 0
 w_k1_INTC_s111_return_constraint: 0.842188 w_k1_INTC_s11 - w_k1_INTC_s111
                                 + 0.99 b_k1_INTC_s111 - 1.01 s_k1_INTC_s111
                                  = 0
 w_k1_INTC_s112_return_constraint: 1.3898 w_k1_INTC_s11 - w_k1_INTC_s112
                                 + 0.99 b_k1_INTC_s112 - 1.01 s_k1_INTC_s112
                                  = 0
 w_k1_INTC_s113_return_constraint: 0.946091 w_k1_INTC_s11 - w_k1_INTC_s113
                                 + 0.99 b_k1_INTC_s113 - 1.01 s_k1_INTC_s113
                                  = 0
 w_k1_INTC_s114_return_constraint: 0.68609 w_k1_INTC_s11 - w_k1_INTC_s114
                                 + 0.99 b_k1_INTC_s114 - 1.01 s_k1_INTC_s114
                                  = 0
 w_k1_INTC_s115_return_constraint: 0.945553 w_k1_INTC_s11 - w_k1_INTC_s115
                                 + 0.99 b_k1_INTC_s115 - 1.01 s_k1_INTC_s115
                                  = 0
 w_k1_INTC_s116_return_constraint: 0.938099 w_k1_INTC_s11 - w_k1_INTC_s116
                                 + 0.99 b_k1_INTC_s116 - 1.01 s_k1_INTC_s116
                                  = 0
 w_k1_INTC_s117_return_constraint: 1.39798 w_k1_INTC_s11 - w_k1_INTC_s117
                                 + 0.99 b_k1_INTC_s117 - 1.01 s_k1_INTC_s117
                                  = 0
 w_k1_KO_s0_return_constraint:   1.15732 w_0_KO - w_k1_KO_s0 + 0.99 b_k1_KO_s0
                                 - 1.01 s_k1_KO_s0  = 0
 w_k1_KO_s1_return_constraint:   0.876229 w_0_KO - w_k1_KO_s1 + 0.99 b_k1_KO_s1
                                 - 1.01 s_k1_KO_s1  = 0
 w_k1_KO_s00_return_constraint:  1.21626 w_k1_KO_s0 - w_k1_KO_s00
                                 + 0.99 b_k1_KO_s00 - 1.01 s_k1_KO_s00  = 0
 w_k1_KO_s01_return_constraint:  0.886292 w_k1_KO_s0 - w_k1_KO_s01
                                 + 0.99 b_k1_KO_s01 - 1.01 s_k1_KO_s01  = 0
 w_k1_KO_s10_return_constraint:  0.817856 w_k1_KO_s1 - w_k1_KO_s10
                                 + 0.99 b_k1_KO_s10 - 1.01 s_k1_KO_s10  = 0
 w_k1_KO_s11_return_constraint:  1.06067 w_k1_KO_s1 - w_k1_KO_s11
                                 + 0.99 b_k1_KO_s11 - 1.01 s_k1_KO_s11  = 0
 w_k1_KO_s000_return_constraint: 1.19122 w_k1_KO_s00 - w_k1_KO_s000
                                 + 0.99 b_k1_KO_s000 - 1.01 s_k1_KO_s000  = 0
 w_k1_KO_s001_return_constraint: 0.755391 w_k1_KO_s00 - w_k1_KO_s001
                                 + 0.99 b_k1_KO_s001 - 1.01 s_k1_KO_s001  = 0
 w_k1_KO_s002_return_constraint: 1.04042 w_k1_KO_s00 - w_k1_KO_s002
                                 + 0.99 b_k1_KO_s002 - 1.01 s_k1_KO_s002  = 0
 w_k1_KO_s003_return_constraint: 0.854356 w_k1_KO_s00 - w_k1_KO_s003
                                 + 0.99 b_k1_KO_s003 - 1.01 s_k1_KO_s003  = 0
 w_k1_KO_s004_return_constraint: 1.03264 w_k1_KO_s00 - w_k1_KO_s004
                                 + 0.99 b_k1_KO_s004 - 1.01 s_k1_KO_s004  = 0
 w_k1_KO_s005_return_constraint: 0.779662 w_k1_KO_s00 - w_k1_KO_s005
                                 + 0.99 b_k1_KO_s005 - 1.01 s_k1_KO_s005  = 0
 w_k1_KO_s006_return_constraint: 0.94526 w_k1_KO_s00 - w_k1_KO_s006
                                 + 0.99 b_k1_KO_s006 - 1.01 s_k1_KO_s006  = 0
 w_k1_KO_s007_return_constraint: 1.18964 w_k1_KO_s00 - w_k1_KO_s007
                                 + 0.99 b_k1_KO_s007 - 1.01 s_k1_KO_s007  = 0
 w_k1_KO_s010_return_constraint: 0.892184 w_k1_KO_s01 - w_k1_KO_s010
                                 + 0.99 b_k1_KO_s010 - 1.01 s_k1_KO_s010  = 0
 w_k1_KO_s011_return_constraint: 0.718984 w_k1_KO_s01 - w_k1_KO_s011
                                 + 0.99 b_k1_KO_s011 - 1.01 s_k1_KO_s011  = 0
 w_k1_KO_s012_return_constraint: 0.971791 w_k1_KO_s01 - w_k1_KO_s012
                                 + 0.99 b_k1_KO_s012 - 1.01 s_k1_KO_s012  = 0
 w_k1_KO_s013_return_constraint: 0.671981 w_k1_KO_s01 - w_k1_KO_s013
                                 + 0.99 b_k1_KO_s013 - 1.01 s_k1_KO_s013  = 0
 w_k1_KO_s014_return_constraint: 1.284 w_k1_KO_s01 - w_k1_KO_s014
                                 + 0.99 b_k1_KO_s014 - 1.01 s_k1_KO_s014  = 0
 w_k1_KO_s015_return_constraint: 1.51998 w_k1_KO_s01 - w_k1_KO_s015
                                 + 0.99 b_k1_KO_s015 - 1.01 s_k1_KO_s015  = 0
 w_k1_KO_s016_return_constraint: 0.869947 w_k1_KO_s01 - w_k1_KO_s016
                                 + 0.99 b_k1_KO_s016 - 1.01 s_k1_KO_s016  = 0
 w_k1_KO_s017_return_constraint: 0.845499 w_k1_KO_s01 - w_k1_KO_s017
                                 + 0.99 b_k1_KO_s017 - 1.01 s_k1_KO_s017  = 0
 w_k1_KO_s100_return_constraint: 0.888832 w_k1_KO_s10 - w_k1_KO_s100
                                 + 0.99 b_k1_KO_s100 - 1.01 s_k1_KO_s100  = 0
 w_k1_KO_s101_return_constraint: 1.09631 w_k1_KO_s10 - w_k1_KO_s101
                                 + 0.99 b_k1_KO_s101 - 1.01 s_k1_KO_s101  = 0
 w_k1_KO_s102_return_constraint: 0.866557 w_k1_KO_s10 - w_k1_KO_s102
                                 + 0.99 b_k1_KO_s102 - 1.01 s_k1_KO_s102  = 0
 w_k1_KO_s103_return_constraint: 0.8049 w_k1_KO_s10 - w_k1_KO_s103
                                 + 0.99 b_k1_KO_s103 - 1.01 s_k1_KO_s103  = 0
 w_k1_KO_s104_return_constraint: 1.1704 w_k1_KO_s10 - w_k1_KO_s104
                                 + 0.99 b_k1_KO_s104 - 1.01 s_k1_KO_s104  = 0
 w_k1_KO_s105_return_constraint: 1.13103 w_k1_KO_s10 - w_k1_KO_s105
                                 + 0.99 b_k1_KO_s105 - 1.01 s_k1_KO_s105  = 0
 w_k1_KO_s106_return_constraint: 1.52796 w_k1_KO_s10 - w_k1_KO_s106
                                 + 0.99 b_k1_KO_s106 - 1.01 s_k1_KO_s106  = 0
 w_k1_KO_s107_return_constraint: 1.23963 w_k1_KO_s10 - w_k1_KO_s107
                                 + 0.99 b_k1_KO_s107 - 1.01 s_k1_KO_s107  = 0
 w_k1_KO_s110_return_constraint: 0.653655 w_k1_KO_s11 - w_k1_KO_s110
                                 + 0.99 b_k1_KO_s110 - 1.01 s_k1_KO_s110  = 0
 w_k1_KO_s111_return_constraint: 1.06847 w_k1_KO_s11 - w_k1_KO_s111
                                 + 0.99 b_k1_KO_s111 - 1.01 s_k1_KO_s111  = 0
 w_k1_KO_s112_return_constraint: 1.22303 w_k1_KO_s11 - w_k1_KO_s112
                                 + 0.99 b_k1_KO_s112 - 1.01 s_k1_KO_s112  = 0
 w_k1_KO_s113_return_constraint: 1.16391 w_k1_KO_s11 - w_k1_KO_s113
                                 + 0.99 b_k1_KO_s113 - 1.01 s_k1_KO_s113  = 0
 w_k1_KO_s114_return_constraint: 0.704768 w_k1_KO_s11 - w_k1_KO_s114
                                 + 0.99 b_k1_KO_s114 - 1.01 s_k1_KO_s114  = 0
 w_k1_KO_s115_return_constraint: 0.921635 w_k1_KO_s11 - w_k1_KO_s115
                                 + 0.99 b_k1_KO_s115 - 1.01 s_k1_KO_s115  = 0
 w_k1_KO_s116_return_constraint: 0.85751 w_k1_KO_s11 - w_k1_KO_s116
                                 + 0.99 b_k1_KO_s116 - 1.01 s_k1_KO_s116  = 0
 w_k1_KO_s117_return_constraint: 1.27891 w_k1_KO_s11 - w_k1_KO_s117
                                 + 0.99 b_k1_KO_s117 - 1.01 s_k1_KO_s117  = 0
 w_k1_MAT_s0_return_constraint:  1.25171 w_0_MAT - w_k1_MAT_s0
                                 + 0.99 b_k1_MAT_s0 - 1.01 s_k1_MAT_s0  = 0
 w_k1_MAT_s1_return_constraint:  0.75446 w_0_MAT - w_k1_MAT_s1
                                 + 0.99 b_k1_MAT_s1 - 1.01 s_k1_MAT_s1  = 0
 w_k1_MAT_s00_return_constraint: 1.3278 w_k1_MAT_s0 - w_k1_MAT_s00
                                 + 0.99 b_k1_MAT_s00 - 1.01 s_k1_MAT_s00  = 0
 w_k1_MAT_s01_return_constraint: 0.92219 w_k1_MAT_s0 - w_k1_MAT_s01
                                 + 0.99 b_k1_MAT_s01 - 1.01 s_k1_MAT_s01  = 0
 w_k1_MAT_s10_return_constraint: 0.797739 w_k1_MAT_s1 - w_k1_MAT_s10
                                 + 0.99 b_k1_MAT_s10 - 1.01 s_k1_MAT_s10  = 0
 w_k1_MAT_s11_return_constraint: 1.22591 w_k1_MAT_s1 - w_k1_MAT_s11
                                 + 0.99 b_k1_MAT_s11 - 1.01 s_k1_MAT_s11  = 0
 w_k1_MAT_s000_return_constraint: 1.15441 w_k1_MAT_s00 - w_k1_MAT_s000
                                 + 0.99 b_k1_MAT_s000 - 1.01 s_k1_MAT_s000  = 0
 w_k1_MAT_s001_return_constraint: 0.68502 w_k1_MAT_s00 - w_k1_MAT_s001
                                 + 0.99 b_k1_MAT_s001 - 1.01 s_k1_MAT_s001  = 0
 w_k1_MAT_s002_return_constraint: 1.26862 w_k1_MAT_s00 - w_k1_MAT_s002
                                 + 0.99 b_k1_MAT_s002 - 1.01 s_k1_MAT_s002  = 0
 w_k1_MAT_s003_return_constraint: 1.00226 w_k1_MAT_s00 - w_k1_MAT_s003
                                 + 0.99 b_k1_MAT_s003 - 1.01 s_k1_MAT_s003  = 0
 w_k1_MAT_s004_return_constraint: 0.690069 w_k1_MAT_s00 - w_k1_MAT_s004
                                 + 0.99 b_k1_MAT_s004 - 1.01 s_k1_MAT_s004  = 0
 w_k1_MAT_s005_return_constraint: 0.681517 w_k1_MAT_s00 - w_k1_MAT_s005
                                 + 0.99 b_k1_MAT_s005 - 1.01 s_k1_MAT_s005  = 0
 w_k1_MAT_s006_return_constraint: 0.921665 w_k1_MAT_s00 - w_k1_MAT_s006
                                 + 0.99 b_k1_MAT_s006 - 1.01 s_k1_MAT_s006  = 0
 w_k1_MAT_s007_return_constraint: 1.20398 w_k1_MAT_s00 - w_k1_MAT_s007
                                 + 0.99 b_k1_MAT_s007 - 1.01 s_k1_MAT_s007  = 0
 w_k1_MAT_s010_return_constraint: 0.662232 w_k1_MAT_s01 - w_k1_MAT_s010
                                 + 0.99 b_k1_MAT_s010 - 1.01 s_k1_MAT_s010  = 0
 w_k1_MAT_s011_return_constraint: 0.519008 w_k1_MAT_s01 - w_k1_MAT_s011
                                 + 0.99 b_k1_MAT_s011 - 1.01 s_k1_MAT_s011  = 0
 w_k1_MAT_s012_return_constraint: 0.897939 w_k1_MAT_s01 - w_k1_MAT_s012
                                 + 0.99 b_k1_MAT_s012 - 1.01 s_k1_MAT_s012  = 0
 w_k1_MAT_s013_return_constraint: 0.441052 w_k1_MAT_s01 - w_k1_MAT_s013
                                 + 0.99 b_k1_MAT_s013 - 1.01 s_k1_MAT_s013  = 0
 w_k1_MAT_s014_return_constraint: 1.10541 w_k1_MAT_s01 - w_k1_MAT_s014
                                 + 0.99 b_k1_MAT_s014 - 1.01 s_k1_MAT_s014  = 0
 w_k1_MAT_s015_return_constraint: 1.58824 w_k1_MAT_s01 - w_k1_MAT_s015
                                 + 0.99 b_k1_MAT_s015 - 1.01 s_k1_MAT_s015  = 0
 w_k1_MAT_s016_return_constraint: 1.12285 w_k1_MAT_s01 - w_k1_MAT_s016
                                 + 0.99 b_k1_MAT_s016 - 1.01 s_k1_MAT_s016  = 0
 w_k1_MAT_s017_return_constraint: 0.8282 w_k1_MAT_s01 - w_k1_MAT_s017
                                 + 0.99 b_k1_MAT_s017 - 1.01 s_k1_MAT_s017  = 0
 w_k1_MAT_s100_return_constraint: 0.707368 w_k1_MAT_s10 - w_k1_MAT_s100
                                 + 0.99 b_k1_MAT_s100 - 1.01 s_k1_MAT_s100  = 0
 w_k1_MAT_s101_return_constraint: 1.07576 w_k1_MAT_s10 - w_k1_MAT_s101
                                 + 0.99 b_k1_MAT_s101 - 1.01 s_k1_MAT_s101  = 0
 w_k1_MAT_s102_return_constraint: 0.980933 w_k1_MAT_s10 - w_k1_MAT_s102
                                 + 0.99 b_k1_MAT_s102 - 1.01 s_k1_MAT_s102  = 0
 w_k1_MAT_s103_return_constraint: 0.774907 w_k1_MAT_s10 - w_k1_MAT_s103
                                 + 0.99 b_k1_MAT_s103 - 1.01 s_k1_MAT_s103  = 0
 w_k1_MAT_s104_return_constraint: 1.08617 w_k1_MAT_s10 - w_k1_MAT_s104
                                 + 0.99 b_k1_MAT_s104 - 1.01 s_k1_MAT_s104  = 0
 w_k1_MAT_s105_return_constraint: 1.50063 w_k1_MAT_s10 - w_k1_MAT_s105
                                 + 0.99 b_k1_MAT_s105 - 1.01 s_k1_MAT_s105  = 0
 w_k1_MAT_s106_return_constraint: 1.28821 w_k1_MAT_s10 - w_k1_MAT_s106
                                 + 0.99 b_k1_MAT_s106 - 1.01 s_k1_MAT_s106  = 0
 w_k1_MAT_s107_return_constraint: 1.29159 w_k1_MAT_s10 - w_k1_MAT_s107
                                 + 0.99 b_k1_MAT_s107 - 1.01 s_k1_MAT_s107  = 0
 w_k1_MAT_s110_return_constraint: 0.642929 w_k1_MAT_s11 - w_k1_MAT_s110
                                 + 0.99 b_k1_MAT_s110 - 1.01 s_k1_MAT_s110  = 0
 w_k1_MAT_s111_return_constraint: 1.35963 w_k1_MAT_s11 - w_k1_MAT_s111
                                 + 0.99 b_k1_MAT_s111 - 1.01 s_k1_MAT_s111  = 0
 w_k1_MAT_s112_return_constraint: 0.965776 w_k1_MAT_s11 - w_k1_MAT_s112
                                 + 0.99 b_k1_MAT_s112 - 1.01 s_k1_MAT_s112  = 0
 w_k1_MAT_s113_return_constraint: 1.20389 w_k1_MAT_s11 - w_k1_MAT_s113
                                 + 0.99 b_k1_MAT_s113 - 1.01 s_k1_MAT_s113  = 0
 w_k1_MAT_s114_return_constraint: 0.695329 w_k1_MAT_s11 - w_k1_MAT_s114
                                 + 0.99 b_k1_MAT_s114 - 1.01 s_k1_MAT_s114  = 0
 w_k1_MAT_s115_return_constraint: 1.07528 w_k1_MAT_s11 - w_k1_MAT_s115
                                 + 0.99 b_k1_MAT_s115 - 1.01 s_k1_MAT_s115  = 0
 w_k1_MAT_s116_return_constraint: 1.04811 w_k1_MAT_s11 - w_k1_MAT_s116
                                 + 0.99 b_k1_MAT_s116 - 1.01 s_k1_MAT_s116  = 0
 w_k1_MAT_s117_return_constraint: 1.58748 w_k1_MAT_s11 - w_k1_MAT_s117
                                 + 0.99 b_k1_MAT_s117 - 1.01 s_k1_MAT_s117  = 0
 w_k1_MSFT_s0_return_constraint: 1.07821 w_0_MSFT - w_k1_MSFT_s0
                                 + 0.99 b_k1_MSFT_s0 - 1.01 s_k1_MSFT_s0  = 0
 w_k1_MSFT_s1_return_constraint: 0.866823 w_0_MSFT - w_k1_MSFT_s1
                                 + 0.99 b_k1_MSFT_s1 - 1.01 s_k1_MSFT_s1  = 0
 w_k1_MSFT_s00_return_constraint: 1.26604 w_k1_MSFT_s0 - w_k1_MSFT_s00
                                 + 0.99 b_k1_MSFT_s00 - 1.01 s_k1_MSFT_s00  = 0
 w_k1_MSFT_s01_return_constraint: 1.0002 w_k1_MSFT_s0 - w_k1_MSFT_s01
                                 + 0.99 b_k1_MSFT_s01 - 1.01 s_k1_MSFT_s01  = 0
 w_k1_MSFT_s10_return_constraint: 0.939661 w_k1_MSFT_s1 - w_k1_MSFT_s10
                                 + 0.99 b_k1_MSFT_s10 - 1.01 s_k1_MSFT_s10  = 0
 w_k1_MSFT_s11_return_constraint: 1.08328 w_k1_MSFT_s1 - w_k1_MSFT_s11
                                 + 0.99 b_k1_MSFT_s11 - 1.01 s_k1_MSFT_s11  = 0
 w_k1_MSFT_s000_return_constraint: 1.19466 w_k1_MSFT_s00 - w_k1_MSFT_s000
                                 + 0.99 b_k1_MSFT_s000 - 1.01 s_k1_MSFT_s000
                                  = 0
 w_k1_MSFT_s001_return_constraint: 0.821757 w_k1_MSFT_s00 - w_k1_MSFT_s001
                                 + 0.99 b_k1_MSFT_s001 - 1.01 s_k1_MSFT_s001
                                  = 0
 w_k1_MSFT_s002_return_constraint: 0.769503 w_k1_MSFT_s00 - w_k1_MSFT_s002
                                 + 0.99 b_k1_MSFT_s002 - 1.01 s_k1_MSFT_s002
                                  = 0
 w_k1_MSFT_s003_return_constraint: 0.664518 w_k1_MSFT_s00 - w_k1_MSFT_s003
                                 + 0.99 b_k1_MSFT_s003 - 1.01 s_k1_MSFT_s003
                                  = 0
 w_k1_MSFT_s004_return_constraint: 1.04661 w_k1_MSFT_s00 - w_k1_MSFT_s004
                                 + 0.99 b_k1_MSFT_s004 - 1.01 s_k1_MSFT_s004
                                  = 0
 w_k1_MSFT_s005_return_constraint: 0.966098 w_k1_MSFT_s00 - w_k1_MSFT_s005
                                 + 0.99 b_k1_MSFT_s005 - 1.01 s_k1_MSFT_s005
                                  = 0
 w_k1_MSFT_s006_return_constraint: 0.903639 w_k1_MSFT_s00 - w_k1_MSFT_s006
                                 + 0.99 b_k1_MSFT_s006 - 1.01 s_k1_MSFT_s006
                                  = 0
 w_k1_MSFT_s007_return_constraint: 1.01444 w_k1_MSFT_s00 - w_k1_MSFT_s007
                                 + 0.99 b_k1_MSFT_s007 - 1.01 s_k1_MSFT_s007
                                  = 0
 w_k1_MSFT_s010_return_constraint: 0.978186 w_k1_MSFT_s01 - w_k1_MSFT_s010
                                 + 0.99 b_k1_MSFT_s010 - 1.01 s_k1_MSFT_s010
                                  = 0
 w_k1_MSFT_s011_return_constraint: 0.677626 w_k1_MSFT_s01 - w_k1_MSFT_s011
                                 + 0.99 b_k1_MSFT_s011 - 1.01 s_k1_MSFT_s011
                                  = 0
 w_k1_MSFT_s012_return_constraint: 0.846378 w_k1_MSFT_s01 - w_k1_MSFT_s012
                                 + 0.99 b_k1_MSFT_s012 - 1.01 s_k1_MSFT_s012
                                  = 0
 w_k1_MSFT_s013_return_constraint: 0.846174 w_k1_MSFT_s01 - w_k1_MSFT_s013
                                 + 0.99 b_k1_MSFT_s013 - 1.01 s_k1_MSFT_s013
                                  = 0
 w_k1_MSFT_s014_return_constraint: 1.11357 w_k1_MSFT_s01 - w_k1_MSFT_s014
                                 + 0.99 b_k1_MSFT_s014 - 1.01 s_k1_MSFT_s014
                                  = 0
 w_k1_MSFT_s015_return_constraint: 1.2534 w_k1_MSFT_s01 - w_k1_MSFT_s015
                                 + 0.99 b_k1_MSFT_s015 - 1.01 s_k1_MSFT_s015
                                  = 0
 w_k1_MSFT_s016_return_constraint: 0.923855 w_k1_MSFT_s01 - w_k1_MSFT_s016
                                 + 0.99 b_k1_MSFT_s016 - 1.01 s_k1_MSFT_s016
                                  = 0
 w_k1_MSFT_s017_return_constraint: 0.683203 w_k1_MSFT_s01 - w_k1_MSFT_s017
                                 + 0.99 b_k1_MSFT_s017 - 1.01 s_k1_MSFT_s017
                                  = 0
 w_k1_MSFT_s100_return_constraint: 1.03567 w_k1_MSFT_s10 - w_k1_MSFT_s100
                                 + 0.99 b_k1_MSFT_s100 - 1.01 s_k1_MSFT_s100
                                  = 0
 w_k1_MSFT_s101_return_constraint: 1.07795 w_k1_MSFT_s10 - w_k1_MSFT_s101
                                 + 0.99 b_k1_MSFT_s101 - 1.01 s_k1_MSFT_s101
                                  = 0
 w_k1_MSFT_s102_return_constraint: 0.895468 w_k1_MSFT_s10 - w_k1_MSFT_s102
                                 + 0.99 b_k1_MSFT_s102 - 1.01 s_k1_MSFT_s102
                                  = 0
 w_k1_MSFT_s103_return_constraint: 0.808458 w_k1_MSFT_s10 - w_k1_MSFT_s103
                                 + 0.99 b_k1_MSFT_s103 - 1.01 s_k1_MSFT_s103
                                  = 0
 w_k1_MSFT_s104_return_constraint: 1.21481 w_k1_MSFT_s10 - w_k1_MSFT_s104
                                 + 0.99 b_k1_MSFT_s104 - 1.01 s_k1_MSFT_s104
                                  = 0
 w_k1_MSFT_s105_return_constraint: 0.962804 w_k1_MSFT_s10 - w_k1_MSFT_s105
                                 + 0.99 b_k1_MSFT_s105 - 1.01 s_k1_MSFT_s105
                                  = 0
 w_k1_MSFT_s106_return_constraint: 1.54559 w_k1_MSFT_s10 - w_k1_MSFT_s106
                                 + 0.99 b_k1_MSFT_s106 - 1.01 s_k1_MSFT_s106
                                  = 0
 w_k1_MSFT_s107_return_constraint: 1.09836 w_k1_MSFT_s10 - w_k1_MSFT_s107
                                 + 0.99 b_k1_MSFT_s107 - 1.01 s_k1_MSFT_s107
                                  = 0
 w_k1_MSFT_s110_return_constraint: 0.907386 w_k1_MSFT_s11 - w_k1_MSFT_s110
                                 + 0.99 b_k1_MSFT_s110 - 1.01 s_k1_MSFT_s110
                                  = 0
 w_k1_MSFT_s111_return_constraint: 0.826698 w_k1_MSFT_s11 - w_k1_MSFT_s111
                                 + 0.99 b_k1_MSFT_s111 - 1.01 s_k1_MSFT_s111
                                  = 0
 w_k1_MSFT_s112_return_constraint: 1.53199 w_k1_MSFT_s11 - w_k1_MSFT_s112
                                 + 0.99 b_k1_MSFT_s112 - 1.01 s_k1_MSFT_s112
                                  = 0
 w_k1_MSFT_s113_return_constraint: 1.00024 w_k1_MSFT_s11 - w_k1_MSFT_s113
                                 + 0.99 b_k1_MSFT_s113 - 1.01 s_k1_MSFT_s113
                                  = 0
 w_k1_MSFT_s114_return_constraint: 0.68973 w_k1_MSFT_s11 - w_k1_MSFT_s114
                                 + 0.99 b_k1_MSFT_s114 - 1.01 s_k1_MSFT_s114
                                  = 0
 w_k1_MSFT_s115_return_constraint: 1.04903 w_k1_MSFT_s11 - w_k1_MSFT_s115
                                 + 0.99 b_k1_MSFT_s115 - 1.01 s_k1_MSFT_s115
                                  = 0
 w_k1_MSFT_s116_return_constraint: 0.841747 w_k1_MSFT_s11 - w_k1_MSFT_s116
                                 + 0.99 b_k1_MSFT_s116 - 1.01 s_k1_MSFT_s116
                                  = 0
 w_k1_MSFT_s117_return_constraint: 1.2749 w_k1_MSFT_s11 - w_k1_MSFT_s117
                                 + 0.99 b_k1_MSFT_s117 - 1.01 s_k1_MSFT_s117
                                  = 0
 w_k1_PG_s0_return_constraint:   1.09025 w_0_PG - w_k1_PG_s0 + 0.99 b_k1_PG_s0
                                 - 1.01 s_k1_PG_s0  = 0
 w_k1_PG_s1_return_constraint:   0.874604 w_0_PG - w_k1_PG_s1 + 0.99 b_k1_PG_s1
                                 - 1.01 s_k1_PG_s1  = 0
 w_k1_PG_s00_return_constraint:  1.18806 w_k1_PG_s0 - w_k1_PG_s00
                                 + 0.99 b_k1_PG_s00 - 1.01 s_k1_PG_s00  = 0
 w_k1_PG_s01_return_constraint:  0.972527 w_k1_PG_s0 - w_k1_PG_s01
                                 + 0.99 b_k1_PG_s01 - 1.01 s_k1_PG_s01  = 0
 w_k1_PG_s10_return_constraint:  0.942813 w_k1_PG_s1 - w_k1_PG_s10
                                 + 0.99 b_k1_PG_s10 - 1.01 s_k1_PG_s10  = 0
 w_k1_PG_s11_return_constraint:  1.11464 w_k1_PG_s1 - w_k1_PG_s11
                                 + 0.99 b_k1_PG_s11 - 1.01 s_k1_PG_s11  = 0
 w_k1_PG_s000_return_constraint: 1.06798 w_k1_PG_s00 - w_k1_PG_s000
                                 + 0.99 b_k1_PG_s000 - 1.01 s_k1_PG_s000  = 0
 w_k1_PG_s001_return_constraint: 0.814867 w_k1_PG_s00 - w_k1_PG_s001
                                 + 0.99 b_k1_PG_s001 - 1.01 s_k1_PG_s001  = 0
 w_k1_PG_s002_return_constraint: 0.927486 w_k1_PG_s00 - w_k1_PG_s002
                                 + 0.99 b_k1_PG_s002 - 1.01 s_k1_PG_s002  = 0
 w_k1_PG_s003_return_constraint: 0.765232 w_k1_PG_s00 - w_k1_PG_s003
                                 + 0.99 b_k1_PG_s003 - 1.01 s_k1_PG_s003  = 0
 w_k1_PG_s004_return_constraint: 1.03453 w_k1_PG_s00 - w_k1_PG_s004
                                 + 0.99 b_k1_PG_s004 - 1.01 s_k1_PG_s004  = 0
 w_k1_PG_s005_return_constraint: 0.927149 w_k1_PG_s00 - w_k1_PG_s005
                                 + 0.99 b_k1_PG_s005 - 1.01 s_k1_PG_s005  = 0
 w_k1_PG_s006_return_constraint: 0.936502 w_k1_PG_s00 - w_k1_PG_s006
                                 + 0.99 b_k1_PG_s006 - 1.01 s_k1_PG_s006  = 0
 w_k1_PG_s007_return_constraint: 1.00918 w_k1_PG_s00 - w_k1_PG_s007
                                 + 0.99 b_k1_PG_s007 - 1.01 s_k1_PG_s007  = 0
 w_k1_PG_s010_return_constraint: 0.929243 w_k1_PG_s01 - w_k1_PG_s010
                                 + 0.99 b_k1_PG_s010 - 1.01 s_k1_PG_s010  = 0
 w_k1_PG_s011_return_constraint: 0.797099 w_k1_PG_s01 - w_k1_PG_s011
                                 + 0.99 b_k1_PG_s011 - 1.01 s_k1_PG_s011  = 0
 w_k1_PG_s012_return_constraint: 0.910526 w_k1_PG_s01 - w_k1_PG_s012
                                 + 0.99 b_k1_PG_s012 - 1.01 s_k1_PG_s012  = 0
 w_k1_PG_s013_return_constraint: 0.8434 w_k1_PG_s01 - w_k1_PG_s013
                                 + 0.99 b_k1_PG_s013 - 1.01 s_k1_PG_s013  = 0
 w_k1_PG_s014_return_constraint: 1.03584 w_k1_PG_s01 - w_k1_PG_s014
                                 + 0.99 b_k1_PG_s014 - 1.01 s_k1_PG_s014  = 0
 w_k1_PG_s015_return_constraint: 1.24123 w_k1_PG_s01 - w_k1_PG_s015
                                 + 0.99 b_k1_PG_s015 - 1.01 s_k1_PG_s015  = 0
 w_k1_PG_s016_return_constraint: 0.911307 w_k1_PG_s01 - w_k1_PG_s016
                                 + 0.99 b_k1_PG_s016 - 1.01 s_k1_PG_s016  = 0
 w_k1_PG_s017_return_constraint: 0.845633 w_k1_PG_s01 - w_k1_PG_s017
                                 + 0.99 b_k1_PG_s017 - 1.01 s_k1_PG_s017  = 0
 w_k1_PG_s100_return_constraint: 1.02469 w_k1_PG_s10 - w_k1_PG_s100
                                 + 0.99 b_k1_PG_s100 - 1.01 s_k1_PG_s100  = 0
 w_k1_PG_s101_return_constraint: 1.01769 w_k1_PG_s10 - w_k1_PG_s101
                                 + 0.99 b_k1_PG_s101 - 1.01 s_k1_PG_s101  = 0
 w_k1_PG_s102_return_constraint: 0.939006 w_k1_PG_s10 - w_k1_PG_s102
                                 + 0.99 b_k1_PG_s102 - 1.01 s_k1_PG_s102  = 0
 w_k1_PG_s103_return_constraint: 0.829376 w_k1_PG_s10 - w_k1_PG_s103
                                 + 0.99 b_k1_PG_s103 - 1.01 s_k1_PG_s103  = 0
 w_k1_PG_s104_return_constraint: 1.14727 w_k1_PG_s10 - w_k1_PG_s104
                                 + 0.99 b_k1_PG_s104 - 1.01 s_k1_PG_s104  = 0
 w_k1_PG_s105_return_constraint: 1.00664 w_k1_PG_s10 - w_k1_PG_s105
                                 + 0.99 b_k1_PG_s105 - 1.01 s_k1_PG_s105  = 0
 w_k1_PG_s106_return_constraint: 1.40261 w_k1_PG_s10 - w_k1_PG_s106
                                 + 0.99 b_k1_PG_s106 - 1.01 s_k1_PG_s106  = 0
 w_k1_PG_s107_return_constraint: 1.12365 w_k1_PG_s10 - w_k1_PG_s107
                                 + 0.99 b_k1_PG_s107 - 1.01 s_k1_PG_s107  = 0
 w_k1_PG_s110_return_constraint: 0.833339 w_k1_PG_s11 - w_k1_PG_s110
                                 + 0.99 b_k1_PG_s110 - 1.01 s_k1_PG_s110  = 0
 w_k1_PG_s111_return_constraint: 0.862655 w_k1_PG_s11 - w_k1_PG_s111
                                 + 0.99 b_k1_PG_s111 - 1.01 s_k1_PG_s111  = 0
 w_k1_PG_s112_return_constraint: 1.34798 w_k1_PG_s11 - w_k1_PG_s112
                                 + 0.99 b_k1_PG_s112 - 1.01 s_k1_PG_s112  = 0
 w_k1_PG_s113_return_constraint: 1.00716 w_k1_PG_s11 - w_k1_PG_s113
                                 + 0.99 b_k1_PG_s113 - 1.01 s_k1_PG_s113  = 0
 w_k1_PG_s114_return_constraint: 0.765323 w_k1_PG_s11 - w_k1_PG_s114
                                 + 0.99 b_k1_PG_s114 - 1.01 s_k1_PG_s114  = 0
 w_k1_PG_s115_return_constraint: 0.972494 w_k1_PG_s11 - w_k1_PG_s115
                                 + 0.99 b_k1_PG_s115 - 1.01 s_k1_PG_s115  = 0
 w_k1_PG_s116_return_constraint: 0.855638 w_k1_PG_s11 - w_k1_PG_s116
                                 + 0.99 b_k1_PG_s116 - 1.01 s_k1_PG_s116  = 0
 w_k1_PG_s117_return_constraint: 1.09851 w_k1_PG_s11 - w_k1_PG_s117
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
 k1_wcvar_constraint:            wcvar - var_1 - 0.248000335798927 z_k1_s0000
                                 - 0.528999950758963 z_k1_s0001
                                 - 0.539998664477077 z_k1_s0002
                                 - 0.669002122665857 z_k1_s0003
                                 - 0.808000132407332 z_k1_s0004
                                 - 0.493001260129789 z_k1_s0005
                                 - 0.999000144288231 z_k1_s0006
                                 - 0.944001345686382 z_k1_s0007
                                 - 0.335000746047058 z_k1_s0010
                                 - 0.0860001093731702 z_k1_s0011
                                 - 0.11400014498304 z_k1_s0012
                                 - 0.291000690088692 z_k1_s0013
                                 - 0.12500009497255 z_k1_s0014
                                 - 0.0830001695579085 z_k1_s0015
                                 - 0.154000835855096 z_k1_s0016
                                 - 0.0920001170038565 z_k1_s0017
                                 - 0.19599949123641 z_k1_s0020
                                 - 0.172999963948702 z_k1_s0021
                                 - 0.166999946482262 z_k1_s0022
                                 - 0.110000104351271 z_k1_s0023
                                 - 0.368000531607995 z_k1_s0024
                                 - 0.210999534902509 z_k1_s0025
                                 - 0.165000480327094 z_k1_s0026
                                 - 0.228999587301828 z_k1_s0027
                                 - 0.141999517070139 z_k1_s0030
                                 - 0.251000369178403 z_k1_s0031
                                 - 0.491999319524828 z_k1_s0032
                                 - 0.269999626568503 z_k1_s0033
                                 - 0.155999569340163 z_k1_s0034
                                 - 0.156999496323945 z_k1_s0035
                                 - 0.174999471428512 z_k1_s0036
                                 - 0.505997867498938 z_k1_s0037
                                 - 0.129000201041455 z_k1_s0040
                                 - 0.149000157858618 z_k1_s0041
                                 - 0.170000097046615 z_k1_s0042
                                 - 0.193000929177738 z_k1_s0043
                                 - 0.0950001816321322 z_k1_s0044
                                 - 0.285001163697369 z_k1_s0045
                                 - 0.154000224413031 z_k1_s0046
                                 - 0.3719999393604 z_k1_s0047
                                 - 0.0900001315351926 z_k1_s0050
                                 - 0.126999973930429 z_k1_s0051
                                 - 0.126999973930429 z_k1_s0052
                                 - 0.200000292300428 z_k1_s0053
                                 - 0.23400085943234 z_k1_s0054
                                 - 0.0810001889417877 z_k1_s0055
                                 - 0.112000187208278 z_k1_s0056
                                 - 0.205000652408511 z_k1_s0057
                                 - 0.131999669439383 z_k1_s0060
                                 - 0.133000655270277 z_k1_s0061
                                 - 0.247000369786108 z_k1_s0062
                                 - 0.181000535066416 z_k1_s0063
                                 - 0.0910001674475705 z_k1_s0064
                                 - 0.185000920383484 z_k1_s0065
                                 - 0.145000625219312 z_k1_s0066
                                 - 0.0720001754948597 z_k1_s0067
                                 - 0.501001513038363 z_k1_s0070
                                 - 0.774999043501421 z_k1_s0071
                                 - 0.721998072997091 z_k1_s0072
                                 - 0.191000050039449 z_k1_s0073
                                 - 0.47000007481963 z_k1_s0074
                                 - 0.546001814712319 z_k1_s0075
                                 - 0.76799961177381 z_k1_s0076
                                 - 0.640999375582316 z_k1_s0077
                                 - 0.803999220276193 z_k1_s0100
                                 - 0.383001220766992 z_k1_s0101
                                 - 0.242999652107455 z_k1_s0102
                                 - 0.862998948634957 z_k1_s0103
                                 - 0.405999560637023 z_k1_s0104
                                 - 0.408999982010816 z_k1_s0105
                                 - 0.331999559407686 z_k1_s0106
                                 - 0.227999746036561 z_k1_s0107
                                 - 0.163999915581479 z_k1_s0110
                                 - 0.322999348593163 z_k1_s0111
                                 - 0.398999700379403 z_k1_s0112
                                 - 0.263999836184842 z_k1_s0113
                                 - 0.421998714762917 z_k1_s0114
                                 - 0.297999797741994 z_k1_s0115
                                 - 0.763000029567168 z_k1_s0116
                                 - 0.228999892593643 z_k1_s0117
                                 - 0.488001252786373 z_k1_s0120
                                 - 1.60700470117332 z_k1_s0121
                                 - 0.462000937610769 z_k1_s0122
                                 - 0.32300069202492 z_k1_s0123
                                 - 0.322000781268374 z_k1_s0124
                                 - 0.967003835864921 z_k1_s0125
                                 - 0.479001001775295 z_k1_s0126
                                 - 0.62300185534604 z_k1_s0127
                                 - 0.374998947123042 z_k1_s0130
                                 - 0.543000036033217 z_k1_s0131
                                 - 0.382999618261246 z_k1_s0132
                                 - 0.308998612229703 z_k1_s0133
                                 - 0.234999947096738 z_k1_s0134
                                 - 0.227999880050494 z_k1_s0135
                                 - 0.344000247960133 z_k1_s0136
                                 - 0.183999830188162 z_k1_s0137
                                 - 2.41899225019858 z_k1_s0140
                                 - 2.33099126456241 z_k1_s0141
                                 - 1.48299802695853 z_k1_s0142
                                 - 0.59599970672584 z_k1_s0143
                                 - 1.26899899828482 z_k1_s0144
                                 - 1.84499278988179 z_k1_s0145
                                 - 0.990998672058206 z_k1_s0146
                                 - 1.8420032093593 z_k1_s0147
                                 - 0.443001005758232 z_k1_s0150
                                 - 0.358999750142002 z_k1_s0151
                                 - 0.428998976739638 z_k1_s0152
                                 - 0.41200000900128 z_k1_s0153
                                 - 0.74699827089627 z_k1_s0154
                                 - 0.321999981447613 z_k1_s0155
                                 - 0.407000091137151 z_k1_s0156
                                 - 0.645998424041521 z_k1_s0157
                                 - 0.440998516381465 z_k1_s0160
                                 - 0.19299992372278 z_k1_s0161
                                 - 0.245999890064273 z_k1_s0162
                                 - 0.551000761051786 z_k1_s0163
                                 - 0.321000032739913 z_k1_s0164
                                 - 0.556000243536988 z_k1_s0165
                                 - 0.285000487647447 z_k1_s0166
                                 - 0.132999864102252 z_k1_s0167
                                 - 0.34899907483927 z_k1_s0170
                                 - 1.26799921207078 z_k1_s0171
                                 - 0.73499900136867 z_k1_s0172
                                 - 0.471998937347843 z_k1_s0173
                                 - 0.517999342288299 z_k1_s0174
                                 - 0.552999755590599 z_k1_s0175
                                 - 0.610999887464145 z_k1_s0176
                                 - 0.332999438928483 z_k1_s0177
                                 - 0.117999654448382 z_k1_s1000
                                 - 0.130999795524533 z_k1_s1001
                                 - 0.393998652134824 z_k1_s1002
                                 - 0.287998992403166 z_k1_s1003
                                 - 0.446998482000653 z_k1_s1004
                                 - 0.167999705834581 z_k1_s1005
                                 - 0.186999624069543 z_k1_s1006
                                 - 0.468998619114347 z_k1_s1007
                                 - 0.146999766893473 z_k1_s1010
                                 - 0.125999800194406 z_k1_s1011
                                 - 0.0809999678963747 z_k1_s1012
                                 - 0.073999950896694 z_k1_s1013
                                 - 0.0370000175983343 z_k1_s1014
                                 - 0.145999969193457 z_k1_s1015
                                 - 0.111000305694927 z_k1_s1016
                                 - 0.120999968694579 z_k1_s1017
                                 - 0.165000043707728 z_k1_s1020
                                 - 0.421000304699247 z_k1_s1021
                                 - 0.280000023334291 z_k1_s1022
                                 - 0.137000041374299 z_k1_s1023
                                 - 0.144000041957656 z_k1_s1024
                                 - 0.265000171870059 z_k1_s1025
                                 - 0.184000135162618 z_k1_s1026
                                 - 0.501000910509462 z_k1_s1027
                                 - 0.200000094811397 z_k1_s1030
                                 - 0.287999918528322 z_k1_s1031
                                 - 0.544000148886955 z_k1_s1032
                                 - 0.362000048983578 z_k1_s1033
                                 - 0.577999987879819 z_k1_s1034
                                 - 0.349999144044524 z_k1_s1035
                                 - 0.158000197526054 z_k1_s1036
                                 - 0.245000218331503 z_k1_s1037
                                 - 0.177000586529057 z_k1_s1040
                                 - 0.201000690329678 z_k1_s1041
                                 - 0.254000883755944 z_k1_s1042
                                 - 0.605001721638037 z_k1_s1043
                                 - 0.233000542329668 z_k1_s1044
                                 - 0.371000781182192 z_k1_s1045
                                 - 0.503000063281837 z_k1_s1046
                                 - 0.520000817009267 z_k1_s1047
                                 - 0.156000590948753 z_k1_s1050
                                 - 0.24699978800117 z_k1_s1051
                                 - 0.188000487450861 z_k1_s1052
                                 - 0.237000055094471 z_k1_s1053
                                 - 0.0880000283810764 z_k1_s1054
                                 - 0.104000101832242 z_k1_s1055
                                 - 0.0750001252021437 z_k1_s1056
                                 - 0.156999687838643 z_k1_s1057
                                 - 0.27800040082567 z_k1_s1060
                                 - 0.121000127363661 z_k1_s1061
                                 - 0.242000521927492 z_k1_s1062
                                 - 0.0870001081375188 z_k1_s1063
                                 - 0.230999890930401 z_k1_s1064
                                 - 0.192000690124467 z_k1_s1065
                                 - 0.0870001081375188 z_k1_s1066
                                 - 0.0980000711341844 z_k1_s1067
                                 - 0.182999333892903 z_k1_s1070
                                 - 0.157000064887388 z_k1_s1071
                                 - 0.356999553299315 z_k1_s1072
                                 - 0.102000088114102 z_k1_s1073
                                 - 0.161000081503628 z_k1_s1074
                                 - 0.427000683083561 z_k1_s1075
                                 - 0.147000023346788 z_k1_s1076
                                 - 0.143999926984604 z_k1_s1077
                                 - 0.116999963103642 z_k1_s1100
                                 - 0.146999896596904 z_k1_s1101
                                 - 0.192000167636659 z_k1_s1102
                                 - 0.302999105802039 z_k1_s1103
                                 - 0.116999963103642 z_k1_s1104
                                 - 0.243999286038604 z_k1_s1105
                                 - 0.512000447031091 z_k1_s1106
                                 - 0.221999730330062 z_k1_s1107
                                 - 1.01200208627271 z_k1_s1110
                                 - 0.476999773152978 z_k1_s1111
                                 - 0.314000388633754 z_k1_s1112
                                 - 0.265000197974212 z_k1_s1113
                                 - 0.325999417039119 z_k1_s1114
                                 - 0.386998960104362 z_k1_s1115
                                 - 0.225000088619532 z_k1_s1116
                                 - 0.234000169924393 z_k1_s1117
                                 - 0.272998907379175 z_k1_s1120
                                 - 0.435000207461048 z_k1_s1121
                                 - 0.430000532168345 z_k1_s1122
                                 - 0.118999933465224 z_k1_s1123
                                 - 0.390999124685827 z_k1_s1124
                                 - 0.192000008537991 z_k1_s1125
                                 - 0.171000058908829 z_k1_s1126
                                 - 0.17800004211855 z_k1_s1127
                                 - 0.758001172983672 z_k1_s1130
                                 - 0.874001964181067 z_k1_s1131
                                 - 0.592001197802988 z_k1_s1132
                                 - 1.05900138524302 z_k1_s1133
                                 - 1.36400372851244 z_k1_s1134
                                 - 0.897002015870042 z_k1_s1135
                                 - 0.418000773608079 z_k1_s1136
                                 - 1.66400340801927 z_k1_s1137
                                 - 0.117999971170195 z_k1_s1140
                                 - 0.173999918652694 z_k1_s1141
                                 - 0.17499975777038 z_k1_s1142
                                 - 0.409998611194186 z_k1_s1143
                                 - 0.4729988905992 z_k1_s1144
                                 - 0.272999822089616 z_k1_s1145
                                 - 0.210999381802289 z_k1_s1146
                                 - 0.248999517268834 z_k1_s1147
                                 - 0.257999071662242 z_k1_s1150
                                 - 0.36700047592153 z_k1_s1151
                                 - 0.162999994700978 z_k1_s1152
                                 - 0.218999761499003 z_k1_s1153
                                 - 0.333000017051673 z_k1_s1154
                                 - 0.166999848215203 z_k1_s1155
                                 - 0.158000007408302 z_k1_s1156
                                 - 0.606998502770816 z_k1_s1157
                                 - 0.370998700239674 z_k1_s1160
                                 - 0.387000230815547 z_k1_s1161
                                 - 0.16899950549453 z_k1_s1162
                                 - 0.572996880088588 z_k1_s1163
                                 - 0.435999420849573 z_k1_s1164
                                 - 0.441997457574947 z_k1_s1165
                                 - 0.571998898794744 z_k1_s1166
                                 - 0.432998710993167 z_k1_s1167
                                 - 0.655000790884384 z_k1_s1170
                                 - 1.07899999293552 z_k1_s1171
                                 - 0.535003111703716 z_k1_s1172
                                 - 0.226000507898682 z_k1_s1173
                                 - 1.08300227084561 z_k1_s1174
                                 - 0.82100195449186 z_k1_s1175
                                 - 0.244000664476653 z_k1_s1176
                                 - 0.404000934723353 z_k1_s1177 >= 0
 z_k1_s0000_constraint:          var_1 + z_k1_s0000 + 0.969365 w_k1_ADM_s000
                                 + 1.53754 w_k1_AXP_s000 + 1.5162 w_k1_DIS_s000
                                 + 1.09619 w_k1_FDX_s000
                                 + 2.29743 w_k1_IBM_s000
                                 + 1.10393 w_k1_INTC_s000
                                 + 1.46021 w_k1_KO_s000 + 1.86045 w_k1_MAT_s000
                                 + 1.22341 w_k1_MSFT_s000
                                 + 1.19277 w_k1_PG_s000 >= 1
 z_k1_s0001_constraint:          var_1 + z_k1_s0001 + 0.932363 w_k1_ADM_s000
                                 + 0.711174 w_k1_AXP_s000
                                 + 0.755171 w_k1_DIS_s000
                                 + 0.775471 w_k1_FDX_s000
                                 + 0.920159 w_k1_IBM_s000
                                 + 0.75796 w_k1_INTC_s000
                                 + 0.890422 w_k1_KO_s000
                                 + 0.851628 w_k1_MAT_s000
                                 + 0.772298 w_k1_MSFT_s000
                                 + 0.826882 w_k1_PG_s000 >= 1
 z_k1_s0002_constraint:          var_1 + z_k1_s0002 + 1.25829 w_k1_ADM_s000
                                 + 1.25572 w_k1_AXP_s000
                                 + 1.62568 w_k1_DIS_s000
                                 + 1.25863 w_k1_FDX_s000
                                 + 1.25765 w_k1_IBM_s000
                                 + 0.918219 w_k1_INTC_s000
                                 + 1.48107 w_k1_KO_s000 + 1.35867 w_k1_MAT_s000
                                 + 1.1077 w_k1_MSFT_s000 + 1.33011 w_k1_PG_s000
                                 >= 1
 z_k1_s0003_constraint:          var_1 + z_k1_s0003 + 1.01193 w_k1_ADM_s000
                                 + 1.04593 w_k1_AXP_s000
                                 + 0.811827 w_k1_DIS_s000
                                 + 0.844506 w_k1_FDX_s000
                                 + 1.11534 w_k1_IBM_s000
                                 + 0.793958 w_k1_INTC_s000
                                 + 1.02442 w_k1_KO_s000 + 1.27832 w_k1_MAT_s000
                                 + 0.871725 w_k1_MSFT_s000
                                 + 0.950694 w_k1_PG_s000 >= 1
 z_k1_s0004_constraint:          var_1 + z_k1_s0004 + 1.31967 w_k1_ADM_s000
                                 + 1.1158 w_k1_AXP_s000 + 0.9822 w_k1_DIS_s000
                                 + 1.14789 w_k1_FDX_s000
                                 + 0.651733 w_k1_IBM_s000
                                 + 0.783179 w_k1_INTC_s000
                                 + 0.753416 w_k1_KO_s000
                                 + 0.652768 w_k1_MAT_s000
                                 + 1.0418 w_k1_MSFT_s000 + 1.0933 w_k1_PG_s000
                                 >= 1
 z_k1_s0005_constraint:          var_1 + z_k1_s0005 + 1.15193 w_k1_ADM_s000
                                 + 1.31112 w_k1_AXP_s000
                                 + 1.62362 w_k1_DIS_s000
                                 + 1.43975 w_k1_FDX_s000
                                 + 0.831973 w_k1_IBM_s000
                                 + 1.16461 w_k1_INTC_s000
                                 + 1.13966 w_k1_KO_s000
                                 + 0.868688 w_k1_MAT_s000
                                 + 1.28276 w_k1_MSFT_s000
                                 + 1.28655 w_k1_PG_s000 >= 1
 z_k1_s0006_constraint:          var_1 + z_k1_s0006 + 0.808287 w_k1_ADM_s000
                                 + 0.78903 w_k1_AXP_s000
                                 + 0.776107 w_k1_DIS_s000
                                 + 0.867833 w_k1_FDX_s000
                                 + 0.634733 w_k1_IBM_s000
                                 + 0.897092 w_k1_INTC_s000
                                 + 0.733672 w_k1_KO_s000
                                 + 0.412688 w_k1_MAT_s000
                                 + 0.931812 w_k1_MSFT_s000
                                 + 0.838566 w_k1_PG_s000 >= 1
 z_k1_s0007_constraint:          var_1 + z_k1_s0007 + 0.817083 w_k1_ADM_s000
                                 + 1.16255 w_k1_AXP_s000
                                 + 1.14149 w_k1_DIS_s000
                                 + 0.994255 w_k1_FDX_s000
                                 + 1.38649 w_k1_IBM_s000
                                 + 1.0363 w_k1_INTC_s000 + 1.24139 w_k1_KO_s000
                                 + 1.44733 w_k1_MAT_s000
                                 + 1.10099 w_k1_MSFT_s000
                                 + 1.18542 w_k1_PG_s000 >= 1
 z_k1_s0010_constraint:          var_1 + z_k1_s0010 + 1.15347 w_k1_ADM_s001
                                 + 1.06368 w_k1_AXP_s001
                                 + 0.995255 w_k1_DIS_s001
                                 + 1.0263 w_k1_FDX_s001
                                 + 0.847602 w_k1_IBM_s001
                                 + 0.945257 w_k1_INTC_s001
                                 + 0.88288 w_k1_KO_s001
                                 + 0.651199 w_k1_MAT_s001
                                 + 0.921677 w_k1_MSFT_s001
                                 + 0.939779 w_k1_PG_s001 >= 1
 z_k1_s0011_constraint:          var_1 + z_k1_s0011 + 1.46244 w_k1_ADM_s001
                                 + 1.09025 w_k1_AXP_s001
                                 + 1.73242 w_k1_DIS_s001
                                 + 1.42599 w_k1_FDX_s001
                                 + 0.953677 w_k1_IBM_s001
                                 + 1.14305 w_k1_INTC_s001
                                 + 1.26188 w_k1_KO_s001
                                 + 0.869578 w_k1_MAT_s001
                                 + 1.367 w_k1_MSFT_s001 + 1.11134 w_k1_PG_s001
                                 >= 1
 z_k1_s0012_constraint:          var_1 + z_k1_s0012 + 1.07352 w_k1_ADM_s001
                                 + 0.921591 w_k1_AXP_s001
                                 + 1.29382 w_k1_DIS_s001
                                 + 1.08993 w_k1_FDX_s001
                                 + 0.909119 w_k1_IBM_s001
                                 + 1.06801 w_k1_INTC_s001
                                 + 1.12646 w_k1_KO_s001
                                 + 0.877535 w_k1_MAT_s001
                                 + 1.19584 w_k1_MSFT_s001
                                 + 1.19067 w_k1_PG_s001 >= 1
 z_k1_s0013_constraint:          var_1 + z_k1_s0013 + 0.830971 w_k1_ADM_s001
                                 + 0.882774 w_k1_AXP_s001
                                 + 0.699252 w_k1_DIS_s001
                                 + 0.836189 w_k1_FDX_s001
                                 + 0.851007 w_k1_IBM_s001
                                 + 0.814452 w_k1_INTC_s001
                                 + 0.79775 w_k1_KO_s001
                                 + 0.804543 w_k1_MAT_s001
                                 + 0.814581 w_k1_MSFT_s001
                                 + 0.860747 w_k1_PG_s001 >= 1
 z_k1_s0014_constraint:          var_1 + z_k1_s0014 + 0.957098 w_k1_ADM_s001
                                 + 1.45203 w_k1_AXP_s001
                                 + 1.53343 w_k1_DIS_s001
                                 + 1.21713 w_k1_FDX_s001
                                 + 1.44242 w_k1_IBM_s001
                                 + 1.12107 w_k1_INTC_s001
                                 + 1.30288 w_k1_KO_s001 + 1.88966 w_k1_MAT_s001
                                 + 1.27135 w_k1_MSFT_s001 + 1.0775 w_k1_PG_s001
                                 >= 1
 z_k1_s0015_constraint:          var_1 + z_k1_s0015 + 0.932569 w_k1_ADM_s001
                                 + 1.25083 w_k1_AXP_s001
                                 + 1.06068 w_k1_DIS_s001
                                 + 1.00416 w_k1_FDX_s001
                                 + 1.43018 w_k1_IBM_s001
                                 + 0.930224 w_k1_INTC_s001
                                 + 1.19097 w_k1_KO_s001 + 1.735 w_k1_MAT_s001
                                 + 0.933138 w_k1_MSFT_s001
                                 + 1.1726 w_k1_PG_s001 >= 1
 z_k1_s0016_constraint:          var_1 + z_k1_s0016 + 1.09599 w_k1_ADM_s001
                                 + 0.883502 w_k1_AXP_s001
                                 + 0.962709 w_k1_DIS_s001
                                 + 0.907765 w_k1_FDX_s001
                                 + 1.12364 w_k1_IBM_s001
                                 + 0.876778 w_k1_INTC_s001
                                 + 1.03768 w_k1_KO_s001
                                 + 0.969939 w_k1_MAT_s001
                                 + 0.855422 w_k1_MSFT_s001
                                 + 0.965766 w_k1_PG_s001 >= 1
 z_k1_s0017_constraint:          var_1 + z_k1_s0017 + 1.40717 w_k1_ADM_s001
                                 + 1.79471 w_k1_AXP_s001
                                 + 1.42121 w_k1_DIS_s001
                                 + 1.35094 w_k1_FDX_s001
                                 + 1.10495 w_k1_IBM_s001
                                 + 1.01096 w_k1_INTC_s001
                                 + 1.24156 w_k1_KO_s001 + 1.34617 w_k1_MAT_s001
                                 + 1.0577 w_k1_MSFT_s001 + 1.28756 w_k1_PG_s001
                                 >= 1
 z_k1_s0020_constraint:          var_1 + z_k1_s0020 + 1.3485 w_k1_ADM_s002
                                 + 1.36954 w_k1_AXP_s002
                                 + 1.35721 w_k1_DIS_s002
                                 + 1.24876 w_k1_FDX_s002
                                 + 1.23078 w_k1_IBM_s002
                                 + 1.33885 w_k1_INTC_s002
                                 + 1.14714 w_k1_KO_s002 + 1.10309 w_k1_MAT_s002
                                 + 1.12838 w_k1_MSFT_s002
                                 + 1.12836 w_k1_PG_s002 >= 1
 z_k1_s0021_constraint:          var_1 + z_k1_s0021 + 1.08931 w_k1_ADM_s002
                                 + 1.23485 w_k1_AXP_s002
                                 + 1.13652 w_k1_DIS_s002 + 1.0197 w_k1_FDX_s002
                                 + 1.52508 w_k1_IBM_s002
                                 + 0.987186 w_k1_INTC_s002
                                 + 1.01991 w_k1_KO_s002 + 1.26608 w_k1_MAT_s002
                                 + 0.90826 w_k1_MSFT_s002
                                 + 1.05387 w_k1_PG_s002 >= 1
 z_k1_s0022_constraint:          var_1 + z_k1_s0022 + 1.00995 w_k1_ADM_s002
                                 + 0.963686 w_k1_AXP_s002
                                 + 0.883663 w_k1_DIS_s002
                                 + 0.834839 w_k1_FDX_s002
                                 + 0.997943 w_k1_IBM_s002
                                 + 1.09696 w_k1_INTC_s002
                                 + 0.95635 w_k1_KO_s002
                                 + 0.954447 w_k1_MAT_s002
                                 + 1.12774 w_k1_MSFT_s002 + 1.0141 w_k1_PG_s002
                                 >= 1
 z_k1_s0023_constraint:          var_1 + z_k1_s0023 + 1.36973 w_k1_ADM_s002
                                 + 1.98802 w_k1_AXP_s002
                                 + 1.60342 w_k1_DIS_s002 + 1.311 w_k1_FDX_s002
                                 + 1.81409 w_k1_IBM_s002
                                 + 1.53797 w_k1_INTC_s002
                                 + 1.26208 w_k1_KO_s002 + 1.3534 w_k1_MAT_s002
                                 + 1.15975 w_k1_MSFT_s002
                                 + 1.06678 w_k1_PG_s002 >= 1
 z_k1_s0024_constraint:          var_1 + z_k1_s0024 + 1.15393 w_k1_ADM_s002
                                 + 1.10742 w_k1_AXP_s002
                                 + 1.06169 w_k1_DIS_s002
                                 + 1.12288 w_k1_FDX_s002
                                 + 0.811861 w_k1_IBM_s002
                                 + 1.33163 w_k1_INTC_s002
                                 + 0.815698 w_k1_KO_s002
                                 + 0.735322 w_k1_MAT_s002
                                 + 1.10361 w_k1_MSFT_s002
                                 + 0.931768 w_k1_PG_s002 >= 1
 z_k1_s0025_constraint:          var_1 + z_k1_s0025 + 0.954934 w_k1_ADM_s002
                                 + 0.947257 w_k1_AXP_s002
                                 + 0.837044 w_k1_DIS_s002
                                 + 0.869762 w_k1_FDX_s002
                                 + 1.17169 w_k1_IBM_s002
                                 + 1.21914 w_k1_INTC_s002
                                 + 0.889821 w_k1_KO_s002
                                 + 0.900646 w_k1_MAT_s002
                                 + 0.914539 w_k1_MSFT_s002
                                 + 0.975529 w_k1_PG_s002 >= 1
 z_k1_s0026_constraint:          var_1 + z_k1_s0026 + 0.841864 w_k1_ADM_s002
                                 + 0.696424 w_k1_AXP_s002
                                 + 0.821053 w_k1_DIS_s002
                                 + 0.884999 w_k1_FDX_s002
                                 + 0.546519 w_k1_IBM_s002
                                 + 1.08405 w_k1_INTC_s002
                                 + 0.637746 w_k1_KO_s002
                                 + 0.490788 w_k1_MAT_s002
                                 + 1.05472 w_k1_MSFT_s002
                                 + 0.868207 w_k1_PG_s002 >= 1
 z_k1_s0027_constraint:          var_1 + z_k1_s0027 + 0.856251 w_k1_ADM_s002
                                 + 0.741235 w_k1_AXP_s002
                                 + 0.784236 w_k1_DIS_s002
                                 + 0.824153 w_k1_FDX_s002
                                 + 0.731838 w_k1_IBM_s002
                                 + 0.948943 w_k1_INTC_s002
                                 + 0.754085 w_k1_KO_s002
                                 + 0.560144 w_k1_MAT_s002
                                 + 0.91538 w_k1_MSFT_s002
                                 + 0.865764 w_k1_PG_s002 >= 1
 z_k1_s0030_constraint:          var_1 + z_k1_s0030 + 1.04512 w_k1_ADM_s003
                                 + 1.46769 w_k1_AXP_s003
                                 + 1.59691 w_k1_DIS_s003
                                 + 1.27191 w_k1_FDX_s003
                                 + 1.11091 w_k1_IBM_s003
                                 + 1.03865 w_k1_INTC_s003
                                 + 1.19665 w_k1_KO_s003 + 1.34209 w_k1_MAT_s003
                                 + 1.19169 w_k1_MSFT_s003
                                 + 1.24929 w_k1_PG_s003 >= 1
 z_k1_s0031_constraint:          var_1 + z_k1_s0031 + 1.01042 w_k1_ADM_s003
                                 + 0.948122 w_k1_AXP_s003
                                 + 0.822017 w_k1_DIS_s003
                                 + 0.939565 w_k1_FDX_s003
                                 + 0.855038 w_k1_IBM_s003
                                 + 0.978696 w_k1_INTC_s003
                                 + 0.859486 w_k1_KO_s003
                                 + 0.82369 w_k1_MAT_s003
                                 + 0.895275 w_k1_MSFT_s003
                                 + 1.05266 w_k1_PG_s003 >= 1
 z_k1_s0032_constraint:          var_1 + z_k1_s0032 + 1.00723 w_k1_ADM_s003
                                 + 0.853977 w_k1_AXP_s003
                                 + 0.870062 w_k1_DIS_s003
                                 + 0.935843 w_k1_FDX_s003
                                 + 0.605989 w_k1_IBM_s003
                                 + 0.87091 w_k1_INTC_s003
                                 + 0.788304 w_k1_KO_s003
                                 + 0.505153 w_k1_MAT_s003
                                 + 0.972878 w_k1_MSFT_s003
                                 + 0.959661 w_k1_PG_s003 >= 1
 z_k1_s0033_constraint:          var_1 + z_k1_s0033 + 1.08271 w_k1_ADM_s003
                                 + 0.940846 w_k1_AXP_s003
                                 + 1.17977 w_k1_DIS_s003
                                 + 1.21444 w_k1_FDX_s003
                                 + 0.775656 w_k1_IBM_s003
                                 + 1.00684 w_k1_INTC_s003
                                 + 1.07204 w_k1_KO_s003
                                 + 0.760313 w_k1_MAT_s003
                                 + 1.29075 w_k1_MSFT_s003
                                 + 1.14164 w_k1_PG_s003 >= 1
 z_k1_s0034_constraint:          var_1 + z_k1_s0034 + 1.23407 w_k1_ADM_s003
                                 + 1.11413 w_k1_AXP_s003 + 1.3494 w_k1_DIS_s003
                                 + 1.16908 w_k1_FDX_s003
                                 + 1.31643 w_k1_IBM_s003
                                 + 1.15189 w_k1_INTC_s003
                                 + 1.22801 w_k1_KO_s003 + 1.06209 w_k1_MAT_s003
                                 + 1.13945 w_k1_MSFT_s003
                                 + 1.21588 w_k1_PG_s003 >= 1
 z_k1_s0035_constraint:          var_1 + z_k1_s0035 + 1.23018 w_k1_ADM_s003
                                 + 1.34638 w_k1_AXP_s003
                                 + 1.37606 w_k1_DIS_s003
                                 + 0.999354 w_k1_FDX_s003
                                 + 2.23712 w_k1_IBM_s003
                                 + 0.97035 w_k1_INTC_s003
                                 + 1.53885 w_k1_KO_s003 + 2.24469 w_k1_MAT_s003
                                 + 1.1886 w_k1_MSFT_s003 + 1.0916 w_k1_PG_s003
                                 >= 1
 z_k1_s0036_constraint:          var_1 + z_k1_s0036 + 0.97434 w_k1_ADM_s003
                                 + 1.1475 w_k1_AXP_s003 + 1.24879 w_k1_DIS_s003
                                 + 1.07102 w_k1_FDX_s003
                                 + 1.58441 w_k1_IBM_s003
                                 + 0.959921 w_k1_INTC_s003
                                 + 1.34661 w_k1_KO_s003 + 1.70018 w_k1_MAT_s003
                                 + 1.14199 w_k1_MSFT_s003
                                 + 1.30301 w_k1_PG_s003 >= 1
 z_k1_s0037_constraint:          var_1 + z_k1_s0037 + 1.00085 w_k1_ADM_s003
                                 + 1.03934 w_k1_AXP_s003
                                 + 0.983585 w_k1_DIS_s003
                                 + 0.972558 w_k1_FDX_s003
                                 + 1.03573 w_k1_IBM_s003
                                 + 1.00253 w_k1_INTC_s003
                                 + 0.939494 w_k1_KO_s003
                                 + 1.03935 w_k1_MAT_s003
                                 + 0.967644 w_k1_MSFT_s003
                                 + 1.06411 w_k1_PG_s003 >= 1
 z_k1_s0040_constraint:          var_1 + z_k1_s0040 + 1.67207 w_k1_ADM_s004
                                 + 1.33512 w_k1_AXP_s004
                                 + 1.62476 w_k1_DIS_s004
                                 + 1.40219 w_k1_FDX_s004
                                 + 1.07499 w_k1_IBM_s004
                                 + 1.1743 w_k1_INTC_s004 + 1.34456 w_k1_KO_s004
                                 + 0.87468 w_k1_MAT_s004
                                 + 1.48198 w_k1_MSFT_s004
                                 + 1.05403 w_k1_PG_s004 >= 1
 z_k1_s0041_constraint:          var_1 + z_k1_s0041 + 1.02859 w_k1_ADM_s004
                                 + 0.876757 w_k1_AXP_s004 + 1.014 w_k1_DIS_s004
                                 + 1.1061 w_k1_FDX_s004
                                 + 0.695713 w_k1_IBM_s004
                                 + 1.15109 w_k1_INTC_s004
                                 + 0.780978 w_k1_KO_s004
                                 + 0.398086 w_k1_MAT_s004
                                 + 1.06098 w_k1_MSFT_s004
                                 + 0.930558 w_k1_PG_s004 >= 1
 z_k1_s0042_constraint:          var_1 + z_k1_s0042 + 0.851131 w_k1_ADM_s004
                                 + 0.686611 w_k1_AXP_s004
                                 + 0.818534 w_k1_DIS_s004
                                 + 0.753631 w_k1_FDX_s004
                                 + 0.775554 w_k1_IBM_s004
                                 + 0.776883 w_k1_INTC_s004
                                 + 0.830533 w_k1_KO_s004
                                 + 0.754629 w_k1_MAT_s004
                                 + 0.852011 w_k1_MSFT_s004
                                 + 0.763208 w_k1_PG_s004 >= 1
 z_k1_s0043_constraint:          var_1 + z_k1_s0043 + 1.16223 w_k1_ADM_s004
                                 + 1.03204 w_k1_AXP_s004
                                 + 0.965907 w_k1_DIS_s004
                                 + 1.00833 w_k1_FDX_s004
                                 + 0.834283 w_k1_IBM_s004
                                 + 0.937142 w_k1_INTC_s004
                                 + 0.889567 w_k1_KO_s004
                                 + 0.828265 w_k1_MAT_s004
                                 + 0.923276 w_k1_MSFT_s004
                                 + 0.924312 w_k1_PG_s004 >= 1
 z_k1_s0044_constraint:          var_1 + z_k1_s0044 + 1.28701 w_k1_ADM_s004
                                 + 1.15759 w_k1_AXP_s004
                                 + 0.780702 w_k1_DIS_s004
                                 + 0.943782 w_k1_FDX_s004
                                 + 1.1362 w_k1_IBM_s004
                                 + 0.750412 w_k1_INTC_s004
                                 + 0.818274 w_k1_KO_s004
                                 + 1.50829 w_k1_MAT_s004
                                 + 0.787972 w_k1_MSFT_s004
                                 + 0.920222 w_k1_PG_s004 >= 1
 z_k1_s0045_constraint:          var_1 + z_k1_s0045 + 1.1448 w_k1_ADM_s004
                                 + 0.934097 w_k1_AXP_s004
                                 + 0.970901 w_k1_DIS_s004
                                 + 0.967349 w_k1_FDX_s004
                                 + 1.21752 w_k1_IBM_s004
                                 + 0.991972 w_k1_INTC_s004
                                 + 1.14614 w_k1_KO_s004 + 1.56765 w_k1_MAT_s004
                                 + 1.06227 w_k1_MSFT_s004
                                 + 1.05353 w_k1_PG_s004 >= 1
 z_k1_s0046_constraint:          var_1 + z_k1_s0046 + 0.920269 w_k1_ADM_s004
                                 + 0.809983 w_k1_AXP_s004
                                 + 0.783851 w_k1_DIS_s004
                                 + 0.863604 w_k1_FDX_s004
                                 + 0.679939 w_k1_IBM_s004
                                 + 0.783342 w_k1_INTC_s004
                                 + 0.74731 w_k1_KO_s004
                                 + 0.805264 w_k1_MAT_s004
                                 + 0.882968 w_k1_MSFT_s004
                                 + 0.874681 w_k1_PG_s004 >= 1
 z_k1_s0047_constraint:          var_1 + z_k1_s0047 + 1.39572 w_k1_ADM_s004
                                 + 1.31851 w_k1_AXP_s004
                                 + 1.20942 w_k1_DIS_s004
                                 + 1.14792 w_k1_FDX_s004
                                 + 1.19831 w_k1_IBM_s004
                                 + 1.07548 w_k1_INTC_s004
                                 + 1.13852 w_k1_KO_s004 + 1.70687 w_k1_MAT_s004
                                 + 1.07668 w_k1_MSFT_s004 + 1.0603 w_k1_PG_s004
                                 >= 1
 z_k1_s0050_constraint:          var_1 + z_k1_s0050 + 1.05011 w_k1_ADM_s005
                                 + 0.767022 w_k1_AXP_s005
                                 + 0.968239 w_k1_DIS_s005
                                 + 0.921083 w_k1_FDX_s005
                                 + 0.682443 w_k1_IBM_s005
                                 + 0.912959 w_k1_INTC_s005
                                 + 0.753746 w_k1_KO_s005
                                 + 0.378227 w_k1_MAT_s005
                                 + 0.95039 w_k1_MSFT_s005
                                 + 0.80469 w_k1_PG_s005 >= 1
 z_k1_s0051_constraint:          var_1 + z_k1_s0051 + 1.18513 w_k1_ADM_s005
                                 + 1.21982 w_k1_AXP_s005
                                 + 1.01608 w_k1_DIS_s005
                                 + 1.12597 w_k1_FDX_s005
                                 + 0.720496 w_k1_IBM_s005
                                 + 0.826989 w_k1_INTC_s005
                                 + 0.808235 w_k1_KO_s005
                                 + 0.857505 w_k1_MAT_s005
                                 + 0.884632 w_k1_MSFT_s005
                                 + 1.08541 w_k1_PG_s005 >= 1
 z_k1_s0052_constraint:          var_1 + z_k1_s0052 + 1.0368 w_k1_ADM_s005
                                 + 1.42983 w_k1_AXP_s005
                                 + 1.49667 w_k1_DIS_s005
                                 + 1.22162 w_k1_FDX_s005
                                 + 1.10311 w_k1_IBM_s005
                                 + 1.13486 w_k1_INTC_s005
                                 + 1.09086 w_k1_KO_s005 + 1.3628 w_k1_MAT_s005
                                 + 1.15769 w_k1_MSFT_s005
                                 + 0.990184 w_k1_PG_s005 >= 1
 z_k1_s0053_constraint:          var_1 + z_k1_s0053 + 0.948053 w_k1_ADM_s005
                                 + 1.00048 w_k1_AXP_s005
                                 + 1.09797 w_k1_DIS_s005
                                 + 1.04781 w_k1_FDX_s005
                                 + 0.928591 w_k1_IBM_s005
                                 + 1.13139 w_k1_INTC_s005
                                 + 1.02479 w_k1_KO_s005
                                 + 0.868915 w_k1_MAT_s005
                                 + 1.12231 w_k1_MSFT_s005
                                 + 0.921802 w_k1_PG_s005 >= 1
 z_k1_s0054_constraint:          var_1 + z_k1_s0054 + 0.807842 w_k1_ADM_s005
                                 + 0.780732 w_k1_AXP_s005
                                 + 0.731529 w_k1_DIS_s005
                                 + 0.7674 w_k1_FDX_s005
                                 + 0.950143 w_k1_IBM_s005
                                 + 0.741508 w_k1_INTC_s005
                                 + 0.916372 w_k1_KO_s005
                                 + 0.975524 w_k1_MAT_s005
                                 + 0.821213 w_k1_MSFT_s005
                                 + 0.796595 w_k1_PG_s005 >= 1
 z_k1_s0055_constraint:          var_1 + z_k1_s0055 + 1.2097 w_k1_ADM_s005
                                 + 1.76087 w_k1_AXP_s005 + 1.1598 w_k1_DIS_s005
                                 + 1.16072 w_k1_FDX_s005 + 1.0548 w_k1_IBM_s005
                                 + 0.957117 w_k1_INTC_s005
                                 + 0.963248 w_k1_KO_s005
                                 + 1.22121 w_k1_MAT_s005
                                 + 1.07367 w_k1_MSFT_s005
                                 + 1.16496 w_k1_PG_s005 >= 1
 z_k1_s0056_constraint:          var_1 + z_k1_s0056 + 1.14027 w_k1_ADM_s005
                                 + 1.4541 w_k1_AXP_s005 + 1.86053 w_k1_DIS_s005
                                 + 1.28578 w_k1_FDX_s005
                                 + 1.19391 w_k1_IBM_s005
                                 + 1.23383 w_k1_INTC_s005
                                 + 1.46405 w_k1_KO_s005 + 1.07787 w_k1_MAT_s005
                                 + 1.28067 w_k1_MSFT_s005
                                 + 1.18368 w_k1_PG_s005 >= 1
 z_k1_s0057_constraint:          var_1 + z_k1_s0057 + 0.945149 w_k1_ADM_s005
                                 + 0.956915 w_k1_AXP_s005
                                 + 0.828932 w_k1_DIS_s005
                                 + 0.900695 w_k1_FDX_s005
                                 + 0.758982 w_k1_IBM_s005
                                 + 0.784244 w_k1_INTC_s005
                                 + 0.761167 w_k1_KO_s005
                                 + 0.592256 w_k1_MAT_s005
                                 + 0.902092 w_k1_MSFT_s005
                                 + 0.857153 w_k1_PG_s005 >= 1
 z_k1_s0060_constraint:          var_1 + z_k1_s0060 + 0.818511 w_k1_ADM_s006
                                 + 0.743704 w_k1_AXP_s006
                                 + 0.728481 w_k1_DIS_s006
                                 + 0.787043 w_k1_FDX_s006
                                 + 0.761798 w_k1_IBM_s006
                                 + 0.838405 w_k1_INTC_s006
                                 + 0.87673 w_k1_KO_s006
                                 + 0.629777 w_k1_MAT_s006
                                 + 0.867384 w_k1_MSFT_s006
                                 + 0.850813 w_k1_PG_s006 >= 1
 z_k1_s0061_constraint:          var_1 + z_k1_s0061 + 0.910876 w_k1_ADM_s006
                                 + 1.54955 w_k1_AXP_s006
                                 + 1.42688 w_k1_DIS_s006
                                 + 1.27538 w_k1_FDX_s006
                                 + 1.42293 w_k1_IBM_s006
                                 + 0.978371 w_k1_INTC_s006
                                 + 1.18058 w_k1_KO_s006 + 1.46834 w_k1_MAT_s006
                                 + 1.22187 w_k1_MSFT_s006
                                 + 1.18303 w_k1_PG_s006 >= 1
 z_k1_s0062_constraint:          var_1 + z_k1_s0062 + 0.871379 w_k1_ADM_s006
                                 + 1.06096 w_k1_AXP_s006
                                 + 0.895935 w_k1_DIS_s006
                                 + 1.03565 w_k1_FDX_s006
                                 + 0.872085 w_k1_IBM_s006
                                 + 0.978717 w_k1_INTC_s006
                                 + 0.7865 w_k1_KO_s006 + 0.897626 w_k1_MAT_s006
                                 + 1.03578 w_k1_MSFT_s006
                                 + 0.957632 w_k1_PG_s006 >= 1
 z_k1_s0063_constraint:          var_1 + z_k1_s0063 + 0.858856 w_k1_ADM_s006
                                 + 0.990696 w_k1_AXP_s006
                                 + 0.737844 w_k1_DIS_s006
                                 + 0.879507 w_k1_FDX_s006
                                 + 1.1816 w_k1_IBM_s006
                                 + 0.883955 w_k1_INTC_s006
                                 + 0.838821 w_k1_KO_s006
                                 + 1.18771 w_k1_MAT_s006
                                 + 0.864873 w_k1_MSFT_s006
                                 + 0.916685 w_k1_PG_s006 >= 1
 z_k1_s0064_constraint:          var_1 + z_k1_s0064 + 1.14315 w_k1_ADM_s006
                                 + 0.912067 w_k1_AXP_s006
                                 + 0.799106 w_k1_DIS_s006
                                 + 0.923184 w_k1_FDX_s006
                                 + 0.980027 w_k1_IBM_s006
                                 + 0.985731 w_k1_INTC_s006
                                 + 0.916955 w_k1_KO_s006
                                 + 0.797334 w_k1_MAT_s006
                                 + 1.04381 w_k1_MSFT_s006 + 1.0697 w_k1_PG_s006
                                 >= 1
 z_k1_s0065_constraint:          var_1 + z_k1_s0065 + 1.00641 w_k1_ADM_s006
                                 + 1.27148 w_k1_AXP_s006
                                 + 1.25477 w_k1_DIS_s006 + 1.1969 w_k1_FDX_s006
                                 + 1.02642 w_k1_IBM_s006
                                 + 1.28342 w_k1_INTC_s006
                                 + 0.994035 w_k1_KO_s006
                                 + 1.12303 w_k1_MAT_s006
                                 + 1.12056 w_k1_MSFT_s006
                                 + 1.11825 w_k1_PG_s006 >= 1
 z_k1_s0066_constraint:          var_1 + z_k1_s0066 + 1.07696 w_k1_ADM_s006
                                 + 1.07415 w_k1_AXP_s006
                                 + 1.10878 w_k1_DIS_s006
                                 + 1.02341 w_k1_FDX_s006 + 1.2354 w_k1_IBM_s006
                                 + 0.929769 w_k1_INTC_s006
                                 + 1.0682 w_k1_KO_s006 + 1.02537 w_k1_MAT_s006
                                 + 1.08902 w_k1_MSFT_s006 + 1.134 w_k1_PG_s006
                                 >= 1
 z_k1_s0067_constraint:          var_1 + z_k1_s0067 + 0.771254 w_k1_ADM_s006
                                 + 0.654743 w_k1_AXP_s006
                                 + 0.608068 w_k1_DIS_s006
                                 + 0.810839 w_k1_FDX_s006
                                 + 0.581832 w_k1_IBM_s006
                                 + 0.939167 w_k1_INTC_s006
                                 + 0.669393 w_k1_KO_s006
                                 + 0.479252 w_k1_MAT_s006
                                 + 0.941945 w_k1_MSFT_s006
                                 + 0.802422 w_k1_PG_s006 >= 1
 z_k1_s0070_constraint:          var_1 + z_k1_s0070 + 1.34991 w_k1_ADM_s007
                                 + 1.66766 w_k1_AXP_s007
                                 + 1.43894 w_k1_DIS_s007
                                 + 1.26186 w_k1_FDX_s007
                                 + 1.81985 w_k1_IBM_s007
                                 + 0.89893 w_k1_INTC_s007
                                 + 1.41893 w_k1_KO_s007 + 1.85538 w_k1_MAT_s007
                                 + 1.06881 w_k1_MSFT_s007
                                 + 1.23038 w_k1_PG_s007 >= 1
 z_k1_s0071_constraint:          var_1 + z_k1_s0071 + 0.939997 w_k1_ADM_s007
                                 + 0.864413 w_k1_AXP_s007
                                 + 0.863021 w_k1_DIS_s007
                                 + 0.915455 w_k1_FDX_s007
                                 + 0.671898 w_k1_IBM_s007
                                 + 0.920416 w_k1_INTC_s007
                                 + 0.749673 w_k1_KO_s007
                                 + 0.57033 w_k1_MAT_s007
                                 + 0.836493 w_k1_MSFT_s007
                                 + 0.846052 w_k1_PG_s007 >= 1
 z_k1_s0072_constraint:          var_1 + z_k1_s0072 + 1.18314 w_k1_ADM_s007
                                 + 1.26783 w_k1_AXP_s007 + 1.1678 w_k1_DIS_s007
                                 + 0.986297 w_k1_FDX_s007
                                 + 1.26391 w_k1_IBM_s007
                                 + 1.10744 w_k1_INTC_s007
                                 + 1.19618 w_k1_KO_s007 + 1.4445 w_k1_MAT_s007
                                 + 0.994996 w_k1_MSFT_s007
                                 + 1.10123 w_k1_PG_s007 >= 1
 z_k1_s0073_constraint:          var_1 + z_k1_s0073 + 1.35407 w_k1_ADM_s007
                                 + 1.72535 w_k1_AXP_s007
                                 + 1.55763 w_k1_DIS_s007
                                 + 1.31774 w_k1_FDX_s007
                                 + 1.14207 w_k1_IBM_s007
                                 + 1.56816 w_k1_INTC_s007
                                 + 1.42172 w_k1_KO_s007 + 1.40738 w_k1_MAT_s007
                                 + 1.45112 w_k1_MSFT_s007
                                 + 1.25707 w_k1_PG_s007 >= 1
 z_k1_s0074_constraint:          var_1 + z_k1_s0074 + 1.02383 w_k1_ADM_s007
                                 + 1.34315 w_k1_AXP_s007
                                 + 0.922277 w_k1_DIS_s007
                                 + 1.00733 w_k1_FDX_s007
                                 + 1.15462 w_k1_IBM_s007
                                 + 1.08185 w_k1_INTC_s007
                                 + 1.05592 w_k1_KO_s007 + 1.65762 w_k1_MAT_s007
                                 + 1.05809 w_k1_MSFT_s007
                                 + 0.987656 w_k1_PG_s007 >= 1
 z_k1_s0075_constraint:          var_1 + z_k1_s0075 + 0.769591 w_k1_ADM_s007
                                 + 0.676322 w_k1_AXP_s007
                                 + 0.621018 w_k1_DIS_s007
                                 + 0.733494 w_k1_FDX_s007
                                 + 0.650123 w_k1_IBM_s007
                                 + 0.817641 w_k1_INTC_s007
                                 + 0.759804 w_k1_KO_s007
                                 + 0.691587 w_k1_MAT_s007
                                 + 0.775769 w_k1_MSFT_s007
                                 + 0.832378 w_k1_PG_s007 >= 1
 z_k1_s0076_constraint:          var_1 + z_k1_s0076 + 1.08902 w_k1_ADM_s007
                                 + 0.865138 w_k1_AXP_s007
                                 + 0.848112 w_k1_DIS_s007
                                 + 0.876961 w_k1_FDX_s007
                                 + 0.936952 w_k1_IBM_s007
                                 + 0.934855 w_k1_INTC_s007
                                 + 0.917497 w_k1_KO_s007
                                 + 0.755502 w_k1_MAT_s007
                                 + 0.81127 w_k1_MSFT_s007
                                 + 0.853838 w_k1_PG_s007 >= 1
 z_k1_s0077_constraint:          var_1 + z_k1_s0077 + 1.27481 w_k1_ADM_s007
                                 + 1.33204 w_k1_AXP_s007
                                 + 1.06159 w_k1_DIS_s007
                                 + 1.08682 w_k1_FDX_s007
                                 + 0.845686 w_k1_IBM_s007
                                 + 0.904439 w_k1_INTC_s007
                                 + 0.861334 w_k1_KO_s007
                                 + 1.05368 w_k1_MAT_s007
                                 + 0.994818 w_k1_MSFT_s007
                                 + 1.11733 w_k1_PG_s007 >= 1
 z_k1_s0100_constraint:          var_1 + z_k1_s0100 + 1.07567 w_k1_ADM_s010
                                 + 1.07952 w_k1_AXP_s010
                                 + 0.890251 w_k1_DIS_s010
                                 + 1.01688 w_k1_FDX_s010
                                 + 0.971005 w_k1_IBM_s010
                                 + 0.980452 w_k1_INTC_s010
                                 + 0.920269 w_k1_KO_s010
                                 + 0.779323 w_k1_MAT_s010
                                 + 0.904062 w_k1_MSFT_s010
                                 + 0.979393 w_k1_PG_s010 >= 1
 z_k1_s0101_constraint:          var_1 + z_k1_s0101 + 1.26761 w_k1_ADM_s010
                                 + 1.32969 w_k1_AXP_s010
                                 + 1.50586 w_k1_DIS_s010
                                 + 1.24355 w_k1_FDX_s010 + 1.4313 w_k1_IBM_s010
                                 + 1.18964 w_k1_INTC_s010
                                 + 1.35678 w_k1_KO_s010 + 1.27961 w_k1_MAT_s010
                                 + 1.41463 w_k1_MSFT_s010
                                 + 1.15364 w_k1_PG_s010 >= 1
 z_k1_s0102_constraint:          var_1 + z_k1_s0102 + 0.835107 w_k1_ADM_s010
                                 + 0.754914 w_k1_AXP_s010
                                 + 0.666218 w_k1_DIS_s010
                                 + 0.709264 w_k1_FDX_s010
                                 + 1.15141 w_k1_IBM_s010
                                 + 0.751473 w_k1_INTC_s010
                                 + 0.832415 w_k1_KO_s010
                                 + 0.773124 w_k1_MAT_s010
                                 + 0.791673 w_k1_MSFT_s010
                                 + 0.805552 w_k1_PG_s010 >= 1
 z_k1_s0103_constraint:          var_1 + z_k1_s0103 + 0.976711 w_k1_ADM_s010
                                 + 0.740192 w_k1_AXP_s010
                                 + 0.703175 w_k1_DIS_s010
                                 + 0.855154 w_k1_FDX_s010
                                 + 0.745746 w_k1_IBM_s010
                                 + 0.836293 w_k1_INTC_s010
                                 + 0.802498 w_k1_KO_s010
                                 + 0.886459 w_k1_MAT_s010
                                 + 0.978998 w_k1_MSFT_s010
                                 + 0.923285 w_k1_PG_s010 >= 1
 z_k1_s0104_constraint:          var_1 + z_k1_s0104 + 0.978656 w_k1_ADM_s010
                                 + 1.18664 w_k1_AXP_s010 + 1.2304 w_k1_DIS_s010
                                 + 1.08334 w_k1_FDX_s010
                                 + 1.40161 w_k1_IBM_s010
                                 + 0.998456 w_k1_INTC_s010
                                 + 1.11027 w_k1_KO_s010 + 1.10797 w_k1_MAT_s010
                                 + 1.09155 w_k1_MSFT_s010
                                 + 1.06652 w_k1_PG_s010 >= 1
 z_k1_s0105_constraint:          var_1 + z_k1_s0105 + 1.16832 w_k1_ADM_s010
                                 + 1.05214 w_k1_AXP_s010 + 1.2394 w_k1_DIS_s010
                                 + 1.15568 w_k1_FDX_s010
                                 + 0.923589 w_k1_IBM_s010
                                 + 0.889303 w_k1_INTC_s010
                                 + 1.10052 w_k1_KO_s010
                                 + 0.980798 w_k1_MAT_s010
                                 + 1.07084 w_k1_MSFT_s010
                                 + 1.01956 w_k1_PG_s010 >= 1
 z_k1_s0106_constraint:          var_1 + z_k1_s0106 + 1.20061 w_k1_ADM_s010
                                 + 1.28313 w_k1_AXP_s010 + 1.2757 w_k1_DIS_s010
                                 + 1.3493 w_k1_FDX_s010
                                 + 0.787323 w_k1_IBM_s010
                                 + 1.30066 w_k1_INTC_s010
                                 + 0.923266 w_k1_KO_s010
                                 + 0.469581 w_k1_MAT_s010
                                 + 1.18628 w_k1_MSFT_s010
                                 + 1.00963 w_k1_PG_s010 >= 1
 z_k1_s0107_constraint:          var_1 + z_k1_s0107 + 0.74918 w_k1_ADM_s010
                                 + 0.629849 w_k1_AXP_s010
                                 + 0.962142 w_k1_DIS_s010
                                 + 0.845623 w_k1_FDX_s010
                                 + 1.41933 w_k1_IBM_s010
                                 + 0.743103 w_k1_INTC_s010
                                 + 1.05346 w_k1_KO_s010
                                 + 0.977717 w_k1_MAT_s010
                                 + 1.01552 w_k1_MSFT_s010
                                 + 0.900354 w_k1_PG_s010 >= 1
 z_k1_s0110_constraint:          var_1 + z_k1_s0110 + 0.975944 w_k1_ADM_s011
                                 + 0.745218 w_k1_AXP_s011
                                 + 1.09798 w_k1_DIS_s011
                                 + 1.06477 w_k1_FDX_s011
                                 + 0.858922 w_k1_IBM_s011
                                 + 0.962869 w_k1_INTC_s011
                                 + 0.885308 w_k1_KO_s011
                                 + 1.02594 w_k1_MAT_s011
                                 + 1.20147 w_k1_MSFT_s011
                                 + 0.968026 w_k1_PG_s011 >= 1
 z_k1_s0111_constraint:          var_1 + z_k1_s0111 + 0.904155 w_k1_ADM_s011
                                 + 1.1377 w_k1_AXP_s011
                                 + 0.830488 w_k1_DIS_s011
                                 + 1.09656 w_k1_FDX_s011
                                 + 0.709721 w_k1_IBM_s011
                                 + 0.902007 w_k1_INTC_s011
                                 + 0.806855 w_k1_KO_s011
                                 + 1.07616 w_k1_MAT_s011
                                 + 0.921837 w_k1_MSFT_s011
                                 + 0.904863 w_k1_PG_s011 >= 1
 z_k1_s0112_constraint:          var_1 + z_k1_s0112 + 0.699211 w_k1_ADM_s011
                                 + 0.844122 w_k1_AXP_s011
                                 + 0.813874 w_k1_DIS_s011
                                 + 0.787003 w_k1_FDX_s011
                                 + 1.16773 w_k1_IBM_s011
                                 + 0.645291 w_k1_INTC_s011
                                 + 0.907616 w_k1_KO_s011
                                 + 1.62662 w_k1_MAT_s011
                                 + 0.757054 w_k1_MSFT_s011
                                 + 0.814423 w_k1_PG_s011 >= 1
 z_k1_s0113_constraint:          var_1 + z_k1_s0113 + 0.705222 w_k1_ADM_s011
                                 + 0.656334 w_k1_AXP_s011
                                 + 0.544233 w_k1_DIS_s011
                                 + 0.739659 w_k1_FDX_s011
                                 + 0.58844 w_k1_IBM_s011
                                 + 0.762846 w_k1_INTC_s011
                                 + 0.657921 w_k1_KO_s011
                                 + 0.618942 w_k1_MAT_s011
                                 + 0.751831 w_k1_MSFT_s011
                                 + 0.682805 w_k1_PG_s011 >= 1
 z_k1_s0114_constraint:          var_1 + z_k1_s0114 + 0.823927 w_k1_ADM_s011
                                 + 0.81579 w_k1_AXP_s011
                                 + 0.806634 w_k1_DIS_s011
                                 + 0.838615 w_k1_FDX_s011
                                 + 0.833995 w_k1_IBM_s011
                                 + 0.800525 w_k1_INTC_s011
                                 + 0.839052 w_k1_KO_s011
                                 + 1.05732 w_k1_MAT_s011
                                 + 0.866083 w_k1_MSFT_s011
                                 + 0.814319 w_k1_PG_s011 >= 1
 z_k1_s0115_constraint:          var_1 + z_k1_s0115 + 1.14946 w_k1_ADM_s011
                                 + 1.24953 w_k1_AXP_s011
                                 + 1.36602 w_k1_DIS_s011
                                 + 1.19782 w_k1_FDX_s011
                                 + 1.67112 w_k1_IBM_s011
                                 + 1.18792 w_k1_INTC_s011
                                 + 1.31521 w_k1_KO_s011 + 1.66066 w_k1_MAT_s011
                                 + 1.09679 w_k1_MSFT_s011
                                 + 1.17555 w_k1_PG_s011 >= 1
 z_k1_s0116_constraint:          var_1 + z_k1_s0116 + 1.2426 w_k1_ADM_s011
                                 + 1.31384 w_k1_AXP_s011
                                 + 1.33631 w_k1_DIS_s011
                                 + 1.24613 w_k1_FDX_s011 + 1.0186 w_k1_IBM_s011
                                 + 1.00292 w_k1_INTC_s011
                                 + 1.05579 w_k1_KO_s011 + 1.41936 w_k1_MAT_s011
                                 + 1.16761 w_k1_MSFT_s011
                                 + 1.06104 w_k1_PG_s011 >= 1
 z_k1_s0117_constraint:          var_1 + z_k1_s0117 + 0.83478 w_k1_ADM_s011
                                 + 0.867994 w_k1_AXP_s011
                                 + 1.09272 w_k1_DIS_s011
                                 + 0.962233 w_k1_FDX_s011
                                 + 1.21105 w_k1_IBM_s011
                                 + 1.08759 w_k1_INTC_s011
                                 + 1.11729 w_k1_KO_s011 + 1.19295 w_k1_MAT_s011
                                 + 1.11411 w_k1_MSFT_s011
                                 + 0.996849 w_k1_PG_s011 >= 1
 z_k1_s0120_constraint:          var_1 + z_k1_s0120 + 0.801574 w_k1_ADM_s012
                                 + 0.825476 w_k1_AXP_s012
                                 + 0.717233 w_k1_DIS_s012
                                 + 0.913767 w_k1_FDX_s012
                                 + 0.650336 w_k1_IBM_s012
                                 + 0.806381 w_k1_INTC_s012
                                 + 0.698166 w_k1_KO_s012
                                 + 0.529348 w_k1_MAT_s012
                                 + 0.836873 w_k1_MSFT_s012
                                 + 0.871287 w_k1_PG_s012 >= 1
 z_k1_s0121_constraint:          var_1 + z_k1_s0121 + 0.864902 w_k1_ADM_s012
                                 + 0.831033 w_k1_AXP_s012
                                 + 1.00172 w_k1_DIS_s012
                                 + 0.941234 w_k1_FDX_s012
                                 + 1.09348 w_k1_IBM_s012
                                 + 0.960163 w_k1_INTC_s012
                                 + 1.08528 w_k1_KO_s012 + 1.12425 w_k1_MAT_s012
                                 + 1.1073 w_k1_MSFT_s012 + 1.09152 w_k1_PG_s012
                                 >= 1
 z_k1_s0122_constraint:          var_1 + z_k1_s0122 + 1.20261 w_k1_ADM_s012
                                 + 1.27769 w_k1_AXP_s012
                                 + 1.39535 w_k1_DIS_s012
                                 + 1.33295 w_k1_FDX_s012
                                 + 0.929129 w_k1_IBM_s012
                                 + 1.22211 w_k1_INTC_s012
                                 + 1.19227 w_k1_KO_s012 + 1.05013 w_k1_MAT_s012
                                 + 1.33876 w_k1_MSFT_s012
                                 + 1.18614 w_k1_PG_s012 >= 1
 z_k1_s0123_constraint:          var_1 + z_k1_s0123 + 1.08855 w_k1_ADM_s012
                                 + 1.40812 w_k1_AXP_s012
                                 + 1.08473 w_k1_DIS_s012
                                 + 1.24366 w_k1_FDX_s012
                                 + 0.727693 w_k1_IBM_s012
                                 + 1.0574 w_k1_INTC_s012
                                 + 0.893332 w_k1_KO_s012
                                 + 0.739302 w_k1_MAT_s012
                                 + 1.08571 w_k1_MSFT_s012
                                 + 1.04453 w_k1_PG_s012 >= 1
 z_k1_s0124_constraint:          var_1 + z_k1_s0124 + 0.963886 w_k1_ADM_s012
                                 + 0.749516 w_k1_AXP_s012
                                 + 0.901725 w_k1_DIS_s012
                                 + 0.868948 w_k1_FDX_s012
                                 + 0.810122 w_k1_IBM_s012
                                 + 0.931585 w_k1_INTC_s012
                                 + 0.890819 w_k1_KO_s012
                                 + 0.520944 w_k1_MAT_s012
                                 + 0.905089 w_k1_MSFT_s012
                                 + 0.864417 w_k1_PG_s012 >= 1
 z_k1_s0125_constraint:          var_1 + z_k1_s0125 + 1.12003 w_k1_ADM_s012
                                 + 1.3661 w_k1_AXP_s012 + 1.31095 w_k1_DIS_s012
                                 + 1.1504 w_k1_FDX_s012 + 1.44817 w_k1_IBM_s012
                                 + 1.02712 w_k1_INTC_s012
                                 + 1.28472 w_k1_KO_s012 + 1.46168 w_k1_MAT_s012
                                 + 1.14122 w_k1_MSFT_s012
                                 + 1.17888 w_k1_PG_s012 >= 1
 z_k1_s0126_constraint:          var_1 + z_k1_s0126 + 0.868578 w_k1_ADM_s012
                                 + 0.886819 w_k1_AXP_s012
                                 + 0.895871 w_k1_DIS_s012
                                 + 1.01933 w_k1_FDX_s012
                                 + 0.861865 w_k1_IBM_s012
                                 + 1.07606 w_k1_INTC_s012
                                 + 0.847105 w_k1_KO_s012
                                 + 0.811921 w_k1_MAT_s012
                                 + 1.12715 w_k1_MSFT_s012
                                 + 0.965603 w_k1_PG_s012 >= 1
 z_k1_s0127_constraint:          var_1 + z_k1_s0127 + 0.92941 w_k1_ADM_s012
                                 + 0.683784 w_k1_AXP_s012
                                 + 0.589724 w_k1_DIS_s012
                                 + 0.702995 w_k1_FDX_s012
                                 + 0.859946 w_k1_IBM_s012
                                 + 0.808044 w_k1_INTC_s012
                                 + 0.842416 w_k1_KO_s012
                                 + 0.739799 w_k1_MAT_s012
                                 + 0.833705 w_k1_MSFT_s012
                                 + 0.853898 w_k1_PG_s012 >= 1
 z_k1_s0130_constraint:          var_1 + z_k1_s0130 + 1.03107 w_k1_ADM_s013
                                 + 1.12976 w_k1_AXP_s013
                                 + 1.02567 w_k1_DIS_s013
                                 + 0.956864 w_k1_FDX_s013
                                 + 1.17588 w_k1_IBM_s013
                                 + 0.884939 w_k1_INTC_s013
                                 + 1.15894 w_k1_KO_s013 + 1.84637 w_k1_MAT_s013
                                 + 0.971659 w_k1_MSFT_s013
                                 + 0.949049 w_k1_PG_s013 >= 1
 z_k1_s0131_constraint:          var_1 + z_k1_s0131 + 0.843701 w_k1_ADM_s013
                                 + 1.00629 w_k1_AXP_s013
                                 + 0.753755 w_k1_DIS_s013
                                 + 0.948576 w_k1_FDX_s013
                                 + 0.883142 w_k1_IBM_s013
                                 + 0.814626 w_k1_INTC_s013
                                 + 0.780423 w_k1_KO_s013
                                 + 0.799139 w_k1_MAT_s013
                                 + 0.924917 w_k1_MSFT_s013
                                 + 0.913758 w_k1_PG_s013 >= 1
 z_k1_s0132_constraint:          var_1 + z_k1_s0132 + 1.3291 w_k1_ADM_s013
                                 + 1.48718 w_k1_AXP_s013
                                 + 1.41557 w_k1_DIS_s013
                                 + 1.25399 w_k1_FDX_s013
                                 + 1.16973 w_k1_IBM_s013
                                 + 1.04212 w_k1_INTC_s013
                                 + 1.31717 w_k1_KO_s013 + 1.98049 w_k1_MAT_s013
                                 + 1.37795 w_k1_MSFT_s013
                                 + 1.17751 w_k1_PG_s013 >= 1
 z_k1_s0133_constraint:          var_1 + z_k1_s0133 + 0.891211 w_k1_ADM_s013
                                 + 0.837449 w_k1_AXP_s013
                                 + 0.793076 w_k1_DIS_s013
                                 + 0.759246 w_k1_FDX_s013
                                 + 1.13214 w_k1_IBM_s013
                                 + 0.739793 w_k1_INTC_s013
                                 + 1.10302 w_k1_KO_s013 + 1.24758 w_k1_MAT_s013
                                 + 0.901529 w_k1_MSFT_s013
                                 + 0.945735 w_k1_PG_s013 >= 1
 z_k1_s0134_constraint:          var_1 + z_k1_s0134 + 1.13594 w_k1_ADM_s013
                                 + 1.20053 w_k1_AXP_s013
                                 + 1.27073 w_k1_DIS_s013
                                 + 1.30481 w_k1_FDX_s013
                                 + 0.830488 w_k1_IBM_s013
                                 + 1.1258 w_k1_INTC_s013 + 1.25274 w_k1_KO_s013
                                 + 1.21463 w_k1_MAT_s013
                                 + 1.25682 w_k1_MSFT_s013
                                 + 1.08796 w_k1_PG_s013 >= 1
 z_k1_s0135_constraint:          var_1 + z_k1_s0135 + 0.695886 w_k1_ADM_s013
                                 + 1.17997 w_k1_AXP_s013
                                 + 1.13485 w_k1_DIS_s013
                                 + 0.936189 w_k1_FDX_s013
                                 + 1.64256 w_k1_IBM_s013
                                 + 1.03418 w_k1_INTC_s013
                                 + 1.24443 w_k1_KO_s013 + 1.46572 w_k1_MAT_s013
                                 + 1.06733 w_k1_MSFT_s013
                                 + 1.01566 w_k1_PG_s013 >= 1
 z_k1_s0136_constraint:          var_1 + z_k1_s0136 + 0.769198 w_k1_ADM_s013
                                 + 0.744717 w_k1_AXP_s013
                                 + 0.606625 w_k1_DIS_s013
                                 + 0.692446 w_k1_FDX_s013
                                 + 0.801484 w_k1_IBM_s013
                                 + 0.827458 w_k1_INTC_s013
                                 + 0.797418 w_k1_KO_s013
                                 + 0.844326 w_k1_MAT_s013
                                 + 0.873103 w_k1_MSFT_s013
                                 + 0.833421 w_k1_PG_s013 >= 1
 z_k1_s0137_constraint:          var_1 + z_k1_s0137 + 0.860325 w_k1_ADM_s013
                                 + 1.07521 w_k1_AXP_s013
                                 + 1.16016 w_k1_DIS_s013
                                 + 1.12305 w_k1_FDX_s013
                                 + 0.806891 w_k1_IBM_s013
                                 + 1.22968 w_k1_INTC_s013
                                 + 0.995626 w_k1_KO_s013
                                 + 0.614565 w_k1_MAT_s013
                                 + 1.16845 w_k1_MSFT_s013
                                 + 0.941404 w_k1_PG_s013 >= 1
 z_k1_s0140_constraint:          var_1 + z_k1_s0140 + 1.0723 w_k1_ADM_s014
                                 + 1.04399 w_k1_AXP_s014
                                 + 0.93501 w_k1_DIS_s014
                                 + 0.901801 w_k1_FDX_s014
                                 + 1.61799 w_k1_IBM_s014
                                 + 0.906468 w_k1_INTC_s014
                                 + 1.10414 w_k1_KO_s014 + 1.51101 w_k1_MAT_s014
                                 + 0.840331 w_k1_MSFT_s014
                                 + 1.06091 w_k1_PG_s014 >= 1
 z_k1_s0141_constraint:          var_1 + z_k1_s0141 + 1.4601 w_k1_ADM_s014
                                 + 1.25821 w_k1_AXP_s014
                                 + 1.49791 w_k1_DIS_s014
                                 + 1.40561 w_k1_FDX_s014
                                 + 1.02501 w_k1_IBM_s014
                                 + 1.08339 w_k1_INTC_s014
                                 + 1.11854 w_k1_KO_s014 + 1.09974 w_k1_MAT_s014
                                 + 1.18003 w_k1_MSFT_s014
                                 + 1.13419 w_k1_PG_s014 >= 1
 z_k1_s0142_constraint:          var_1 + z_k1_s0142 + 0.880837 w_k1_ADM_s014
                                 + 0.640814 w_k1_AXP_s014
                                 + 0.612214 w_k1_DIS_s014
                                 + 0.720778 w_k1_FDX_s014
                                 + 0.854542 w_k1_IBM_s014
                                 + 0.592741 w_k1_INTC_s014
                                 + 0.642131 w_k1_KO_s014
                                 + 0.808018 w_k1_MAT_s014
                                 + 0.663698 w_k1_MSFT_s014
                                 + 0.77685 w_k1_PG_s014 >= 1
 z_k1_s0143_constraint:          var_1 + z_k1_s0143 + 0.87835 w_k1_ADM_s014
                                 + 0.793895 w_k1_AXP_s014
                                 + 0.770995 w_k1_DIS_s014
                                 + 0.951144 w_k1_FDX_s014
                                 + 0.493399 w_k1_IBM_s014
                                 + 0.809018 w_k1_INTC_s014
                                 + 0.694406 w_k1_KO_s014
                                 + 0.632759 w_k1_MAT_s014
                                 + 0.879404 w_k1_MSFT_s014
                                 + 0.814285 w_k1_PG_s014 >= 1
 z_k1_s0144_constraint:          var_1 + z_k1_s0144 + 1.01727 w_k1_ADM_s014
                                 + 1.03296 w_k1_AXP_s014
                                 + 0.871611 w_k1_DIS_s014
                                 + 1.04379 w_k1_FDX_s014
                                 + 0.694964 w_k1_IBM_s014
                                 + 0.889801 w_k1_INTC_s014
                                 + 0.779114 w_k1_KO_s014
                                 + 0.828561 w_k1_MAT_s014
                                 + 0.903111 w_k1_MSFT_s014
                                 + 0.958849 w_k1_PG_s014 >= 1
 z_k1_s0145_constraint:          var_1 + z_k1_s0145 + 0.969775 w_k1_ADM_s014
                                 + 0.954883 w_k1_AXP_s014
                                 + 0.761806 w_k1_DIS_s014
                                 + 0.869439 w_k1_FDX_s014
                                 + 1.07922 w_k1_IBM_s014
                                 + 0.691834 w_k1_INTC_s014
                                 + 0.874669 w_k1_KO_s014
                                 + 1.18034 w_k1_MAT_s014
                                 + 0.779724 w_k1_MSFT_s014
                                 + 0.944764 w_k1_PG_s014 >= 1
 z_k1_s0146_constraint:          var_1 + z_k1_s0146 + 1.08939 w_k1_ADM_s014
                                 + 0.849168 w_k1_AXP_s014
                                 + 1.1913 w_k1_DIS_s014 + 1.0467 w_k1_FDX_s014
                                 + 0.999335 w_k1_IBM_s014
                                 + 0.860963 w_k1_INTC_s014
                                 + 0.962837 w_k1_KO_s014
                                 + 1.12232 w_k1_MAT_s014
                                 + 1.05059 w_k1_MSFT_s014
                                 + 1.10117 w_k1_PG_s014 >= 1
 z_k1_s0147_constraint:          var_1 + z_k1_s0147 + 1.09842 w_k1_ADM_s014
                                 + 1.00982 w_k1_AXP_s014
                                 + 1.01568 w_k1_DIS_s014
                                 + 1.21335 w_k1_FDX_s014
                                 + 0.767508 w_k1_IBM_s014
                                 + 0.908546 w_k1_INTC_s014
                                 + 0.910223 w_k1_KO_s014
                                 + 0.904705 w_k1_MAT_s014
                                 + 1.10226 w_k1_MSFT_s014
                                 + 1.14518 w_k1_PG_s014 >= 1
 z_k1_s0150_constraint:          var_1 + z_k1_s0150 + 0.935062 w_k1_ADM_s015
                                 + 0.883885 w_k1_AXP_s015
                                 + 0.95654 w_k1_DIS_s015
                                 + 1.01571 w_k1_FDX_s015
                                 + 0.683989 w_k1_IBM_s015
                                 + 1.10695 w_k1_INTC_s015
                                 + 0.745726 w_k1_KO_s015
                                 + 0.550254 w_k1_MAT_s015
                                 + 1.02995 w_k1_MSFT_s015
                                 + 0.948482 w_k1_PG_s015 >= 1
 z_k1_s0151_constraint:          var_1 + z_k1_s0151 + 0.9767 w_k1_ADM_s015
                                 + 0.941761 w_k1_AXP_s015
                                 + 0.694675 w_k1_DIS_s015
                                 + 0.911217 w_k1_FDX_s015
                                 + 0.568625 w_k1_IBM_s015
                                 + 0.933105 w_k1_INTC_s015
                                 + 0.661324 w_k1_KO_s015
                                 + 0.606817 w_k1_MAT_s015
                                 + 0.844458 w_k1_MSFT_s015
                                 + 0.932941 w_k1_PG_s015 >= 1
 z_k1_s0152_constraint:          var_1 + z_k1_s0152 + 0.784155 w_k1_ADM_s015
                                 + 1.17588 w_k1_AXP_s015
                                 + 1.58552 w_k1_DIS_s015
                                 + 1.17762 w_k1_FDX_s015
                                 + 1.72949 w_k1_IBM_s015
                                 + 1.29027 w_k1_INTC_s015
                                 + 1.43126 w_k1_KO_s015 + 1.68753 w_k1_MAT_s015
                                 + 1.30935 w_k1_MSFT_s015
                                 + 1.18397 w_k1_PG_s015 >= 1
 z_k1_s0153_constraint:          var_1 + z_k1_s0153 + 1.01947 w_k1_ADM_s015
                                 + 1.0156 w_k1_AXP_s015 + 1.37218 w_k1_DIS_s015
                                 + 1.3857 w_k1_FDX_s015
                                 + 0.610759 w_k1_IBM_s015
                                 + 1.28606 w_k1_INTC_s015
                                 + 0.844396 w_k1_KO_s015
                                 + 0.431594 w_k1_MAT_s015
                                 + 1.41814 w_k1_MSFT_s015
                                 + 1.03453 w_k1_PG_s015 >= 1
 z_k1_s0154_constraint:          var_1 + z_k1_s0154 + 0.792958 w_k1_ADM_s015
                                 + 0.869853 w_k1_AXP_s015
                                 + 0.785494 w_k1_DIS_s015
                                 + 0.795053 w_k1_FDX_s015
                                 + 1.31419 w_k1_IBM_s015
                                 + 0.916587 w_k1_INTC_s015
                                 + 1.04757 w_k1_KO_s015
                                 + 0.996375 w_k1_MAT_s015
                                 + 0.913062 w_k1_MSFT_s015
                                 + 0.906999 w_k1_PG_s015 >= 1
 z_k1_s0155_constraint:          var_1 + z_k1_s0155 + 0.810707 w_k1_ADM_s015
                                 + 0.605884 w_k1_AXP_s015
                                 + 0.620006 w_k1_DIS_s015
                                 + 0.729805 w_k1_FDX_s015
                                 + 0.763617 w_k1_IBM_s015
                                 + 0.899415 w_k1_INTC_s015
                                 + 0.729105 w_k1_KO_s015
                                 + 0.662419 w_k1_MAT_s015
                                 + 0.923946 w_k1_MSFT_s015
                                 + 0.808062 w_k1_PG_s015 >= 1
 z_k1_s0156_constraint:          var_1 + z_k1_s0156 + 0.893597 w_k1_ADM_s015
                                 + 0.905819 w_k1_AXP_s015
                                 + 1.20387 w_k1_DIS_s015
                                 + 0.974906 w_k1_FDX_s015
                                 + 1.18465 w_k1_IBM_s015
                                 + 1.20363 w_k1_INTC_s015
                                 + 1.21545 w_k1_KO_s015 + 1.27039 w_k1_MAT_s015
                                 + 1.2137 w_k1_MSFT_s015 + 1.03689 w_k1_PG_s015
                                 >= 1
 z_k1_s0157_constraint:          var_1 + z_k1_s0157 + 1.34694 w_k1_ADM_s015
                                 + 1.04338 w_k1_AXP_s015
                                 + 1.04587 w_k1_DIS_s015
                                 + 1.10419 w_k1_FDX_s015
                                 + 0.979411 w_k1_IBM_s015
                                 + 1.04673 w_k1_INTC_s015
                                 + 1.02416 w_k1_KO_s015
                                 + 0.780085 w_k1_MAT_s015
                                 + 1.11552 w_k1_MSFT_s015 + 1.0619 w_k1_PG_s015
                                 >= 1
 z_k1_s0160_constraint:          var_1 + z_k1_s0160 + 1.00532 w_k1_ADM_s016
                                 + 0.972067 w_k1_AXP_s016
                                 + 0.911934 w_k1_DIS_s016
                                 + 1.02484 w_k1_FDX_s016
                                 + 0.704865 w_k1_IBM_s016
                                 + 1.15653 w_k1_INTC_s016
                                 + 0.980256 w_k1_KO_s016
                                 + 0.700616 w_k1_MAT_s016
                                 + 1.02052 w_k1_MSFT_s016
                                 + 1.01884 w_k1_PG_s016 >= 1
 z_k1_s0161_constraint:          var_1 + z_k1_s0161 + 0.993309 w_k1_ADM_s016
                                 + 1.20464 w_k1_AXP_s016
                                 + 1.26252 w_k1_DIS_s016
                                 + 1.17564 w_k1_FDX_s016
                                 + 1.18841 w_k1_IBM_s016
                                 + 1.1849 w_k1_INTC_s016 + 1.15668 w_k1_KO_s016
                                 + 0.946102 w_k1_MAT_s016
                                 + 1.11971 w_k1_MSFT_s016
                                 + 1.22221 w_k1_PG_s016 >= 1
 z_k1_s0162_constraint:          var_1 + z_k1_s0162 + 1.21858 w_k1_ADM_s016
                                 + 0.96923 w_k1_AXP_s016
                                 + 1.06942 w_k1_DIS_s016
                                 + 1.23493 w_k1_FDX_s016
                                 + 0.662167 w_k1_IBM_s016
                                 + 1.29809 w_k1_INTC_s016
                                 + 1.06039 w_k1_KO_s016
                                 + 0.608479 w_k1_MAT_s016
                                 + 1.16731 w_k1_MSFT_s016 + 1.0895 w_k1_PG_s016
                                 >= 1
 z_k1_s0163_constraint:          var_1 + z_k1_s0163 + 0.855165 w_k1_ADM_s016
                                 + 0.935554 w_k1_AXP_s016
                                 + 0.807929 w_k1_DIS_s016
                                 + 0.911184 w_k1_FDX_s016
                                 + 1.15654 w_k1_IBM_s016
                                 + 0.875642 w_k1_INTC_s016
                                 + 1.08134 w_k1_KO_s016 + 1.12905 w_k1_MAT_s016
                                 + 0.80985 w_k1_MSFT_s016
                                 + 1.04558 w_k1_PG_s016 >= 1
 z_k1_s0164_constraint:          var_1 + z_k1_s0164 + 0.956736 w_k1_ADM_s016
                                 + 1.2008 w_k1_AXP_s016 + 1.1527 w_k1_DIS_s016
                                 + 1.07978 w_k1_FDX_s016 + 1.5232 w_k1_IBM_s016
                                 + 1.07521 w_k1_INTC_s016
                                 + 1.40425 w_k1_KO_s016 + 1.68711 w_k1_MAT_s016
                                 + 1.24057 w_k1_MSFT_s016
                                 + 1.13664 w_k1_PG_s016 >= 1
 z_k1_s0165_constraint:          var_1 + z_k1_s0165 + 0.730654 w_k1_ADM_s016
                                 + 0.634217 w_k1_AXP_s016
                                 + 0.642039 w_k1_DIS_s016
                                 + 0.764699 w_k1_FDX_s016
                                 + 0.908292 w_k1_IBM_s016
                                 + 1.05402 w_k1_INTC_s016
                                 + 0.909925 w_k1_KO_s016
                                 + 0.754272 w_k1_MAT_s016
                                 + 0.895403 w_k1_MSFT_s016
                                 + 0.91623 w_k1_PG_s016 >= 1
 z_k1_s0166_constraint:          var_1 + z_k1_s0166 + 1.34725 w_k1_ADM_s016
                                 + 1.26313 w_k1_AXP_s016
                                 + 1.47958 w_k1_DIS_s016
                                 + 1.42699 w_k1_FDX_s016 + 0.8169 w_k1_IBM_s016
                                 + 1.1268 w_k1_INTC_s016 + 1.09815 w_k1_KO_s016
                                 + 1.18857 w_k1_MAT_s016
                                 + 1.23213 w_k1_MSFT_s016
                                 + 1.19134 w_k1_PG_s016 >= 1
 z_k1_s0167_constraint:          var_1 + z_k1_s0167 + 0.953149 w_k1_ADM_s016
                                 + 0.942176 w_k1_AXP_s016
                                 + 0.810586 w_k1_DIS_s016
                                 + 0.779636 w_k1_FDX_s016
                                 + 1.90933 w_k1_IBM_s016
                                 + 0.852941 w_k1_INTC_s016
                                 + 1.19837 w_k1_KO_s016 + 1.70331 w_k1_MAT_s016
                                 + 0.862902 w_k1_MSFT_s016
                                 + 1.02529 w_k1_PG_s016 >= 1
 z_k1_s0170_constraint:          var_1 + z_k1_s0170 + 1.2137 w_k1_ADM_s017
                                 + 1.35769 w_k1_AXP_s017
                                 + 1.38356 w_k1_DIS_s017
                                 + 1.17923 w_k1_FDX_s017
                                 + 1.80474 w_k1_IBM_s017
                                 + 1.2711 w_k1_INTC_s017 + 1.61838 w_k1_KO_s017
                                 + 2.29162 w_k1_MAT_s017
                                 + 1.46362 w_k1_MSFT_s017
                                 + 1.29598 w_k1_PG_s017 >= 1
 z_k1_s0171_constraint:          var_1 + z_k1_s0171 + 1.12441 w_k1_ADM_s017
                                 + 0.863491 w_k1_AXP_s017
                                 + 0.933868 w_k1_DIS_s017
                                 + 0.983825 w_k1_FDX_s017
                                 + 0.961484 w_k1_IBM_s017
                                 + 0.884957 w_k1_INTC_s017
                                 + 1.0678 w_k1_KO_s017 + 1.14628 w_k1_MAT_s017
                                 + 1.14405 w_k1_MSFT_s017
                                 + 0.979575 w_k1_PG_s017 >= 1
 z_k1_s0172_constraint:          var_1 + z_k1_s0172 + 0.996498 w_k1_ADM_s017
                                 + 1.0351 w_k1_AXP_s017 + 0.9911 w_k1_DIS_s017
                                 + 1.11973 w_k1_FDX_s017
                                 + 0.670073 w_k1_IBM_s017
                                 + 1.08059 w_k1_INTC_s017
                                 + 0.971473 w_k1_KO_s017
                                 + 1.04381 w_k1_MAT_s017
                                 + 1.14566 w_k1_MSFT_s017
                                 + 1.05146 w_k1_PG_s017 >= 1
 z_k1_s0173_constraint:          var_1 + z_k1_s0173 + 0.840819 w_k1_ADM_s017
                                 + 0.878077 w_k1_AXP_s017
                                 + 1.03889 w_k1_DIS_s017
                                 + 0.879521 w_k1_FDX_s017
                                 + 1.31034 w_k1_IBM_s017
                                 + 0.953308 w_k1_INTC_s017
                                 + 1.08352 w_k1_KO_s017 + 1.44361 w_k1_MAT_s017
                                 + 1.00174 w_k1_MSFT_s017
                                 + 1.00375 w_k1_PG_s017 >= 1
 z_k1_s0174_constraint:          var_1 + z_k1_s0174 + 1.10439 w_k1_ADM_s017
                                 + 0.950813 w_k1_AXP_s017
                                 + 1.48872 w_k1_DIS_s017
                                 + 1.20357 w_k1_FDX_s017
                                 + 1.01428 w_k1_IBM_s017
                                 + 0.949205 w_k1_INTC_s017
                                 + 1.31473 w_k1_KO_s017 + 1.08557 w_k1_MAT_s017
                                 + 1.36441 w_k1_MSFT_s017
                                 + 1.08081 w_k1_PG_s017 >= 1
 z_k1_s0175_constraint:          var_1 + z_k1_s0175 + 0.992529 w_k1_ADM_s017
                                 + 1.27482 w_k1_AXP_s017
                                 + 1.11491 w_k1_DIS_s017
                                 + 1.12276 w_k1_FDX_s017
                                 + 1.39044 w_k1_IBM_s017
                                 + 1.05262 w_k1_INTC_s017
                                 + 1.14899 w_k1_KO_s017 + 1.59005 w_k1_MAT_s017
                                 + 1.2429 w_k1_MSFT_s017 + 1.07618 w_k1_PG_s017
                                 >= 1
 z_k1_s0176_constraint:          var_1 + z_k1_s0176 + 0.685558 w_k1_ADM_s017
                                 + 0.643686 w_k1_AXP_s017
                                 + 0.633095 w_k1_DIS_s017
                                 + 0.767652 w_k1_FDX_s017
                                 + 0.720301 w_k1_IBM_s017
                                 + 0.752337 w_k1_INTC_s017
                                 + 0.721662 w_k1_KO_s017
                                 + 0.671187 w_k1_MAT_s017
                                 + 0.814382 w_k1_MSFT_s017
                                 + 0.701415 w_k1_PG_s017 >= 1
 z_k1_s0177_constraint:          var_1 + z_k1_s0177 + 0.84153 w_k1_ADM_s017
                                 + 0.680104 w_k1_AXP_s017
                                 + 0.707612 w_k1_DIS_s017
                                 + 0.705144 w_k1_FDX_s017
                                 + 1.20837 w_k1_IBM_s017
                                 + 0.76274 w_k1_INTC_s017
                                 + 0.887631 w_k1_KO_s017
                                 + 0.960587 w_k1_MAT_s017
                                 + 0.800886 w_k1_MSFT_s017
                                 + 0.76475 w_k1_PG_s017 >= 1
 z_k1_s1000_constraint:          var_1 + z_k1_s1000 + 0.994056 w_k1_ADM_s100
                                 + 1.28147 w_k1_AXP_s100
                                 + 1.27273 w_k1_DIS_s100
                                 + 0.991867 w_k1_FDX_s100
                                 + 1.80477 w_k1_IBM_s100
                                 + 1.20192 w_k1_INTC_s100
                                 + 1.68469 w_k1_KO_s100 + 2.4146 w_k1_MAT_s100
                                 + 1.1584 w_k1_MSFT_s100 + 1.17309 w_k1_PG_s100
                                 >= 1
 z_k1_s1001_constraint:          var_1 + z_k1_s1001 + 0.948646 w_k1_ADM_s100
                                 + 0.807768 w_k1_AXP_s100
                                 + 0.778136 w_k1_DIS_s100
                                 + 0.973069 w_k1_FDX_s100
                                 + 0.597841 w_k1_IBM_s100
                                 + 0.918856 w_k1_INTC_s100
                                 + 0.761351 w_k1_KO_s100
                                 + 0.547995 w_k1_MAT_s100
                                 + 1.01649 w_k1_MSFT_s100
                                 + 0.797076 w_k1_PG_s100 >= 1
 z_k1_s1002_constraint:          var_1 + z_k1_s1002 + 1.01401 w_k1_ADM_s100
                                 + 1.0734 w_k1_AXP_s100 + 1.44389 w_k1_DIS_s100
                                 + 1.07001 w_k1_FDX_s100
                                 + 1.19552 w_k1_IBM_s100
                                 + 1.18362 w_k1_INTC_s100
                                 + 1.31754 w_k1_KO_s100 + 1.35238 w_k1_MAT_s100
                                 + 1.25104 w_k1_MSFT_s100
                                 + 1.06216 w_k1_PG_s100 >= 1
 z_k1_s1003_constraint:          var_1 + z_k1_s1003 + 1.19022 w_k1_ADM_s100
                                 + 1.17772 w_k1_AXP_s100
                                 + 1.22079 w_k1_DIS_s100
                                 + 1.23894 w_k1_FDX_s100
                                 + 0.885196 w_k1_IBM_s100
                                 + 0.986353 w_k1_INTC_s100
                                 + 1.0196 w_k1_KO_s100 + 0.986179 w_k1_MAT_s100
                                 + 1.11521 w_k1_MSFT_s100
                                 + 0.982521 w_k1_PG_s100 >= 1
 z_k1_s1004_constraint:          var_1 + z_k1_s1004 + 1.03732 w_k1_ADM_s100
                                 + 0.859576 w_k1_AXP_s100
                                 + 0.780019 w_k1_DIS_s100
                                 + 0.812917 w_k1_FDX_s100
                                 + 1.10931 w_k1_IBM_s100
                                 + 0.850495 w_k1_INTC_s100
                                 + 0.956477 w_k1_KO_s100
                                 + 0.970401 w_k1_MAT_s100
                                 + 0.94768 w_k1_MSFT_s100
                                 + 0.81955 w_k1_PG_s100 >= 1
 z_k1_s1005_constraint:          var_1 + z_k1_s1005 + 1.16529 w_k1_ADM_s100
                                 + 1.5507 w_k1_AXP_s100 + 1.59034 w_k1_DIS_s100
                                 + 1.44592 w_k1_FDX_s100
                                 + 1.18847 w_k1_IBM_s100
                                 + 1.38235 w_k1_INTC_s100
                                 + 1.27249 w_k1_KO_s100 + 1.95397 w_k1_MAT_s100
                                 + 1.54276 w_k1_MSFT_s100
                                 + 1.10057 w_k1_PG_s100 >= 1
 z_k1_s1006_constraint:          var_1 + z_k1_s1006 + 0.682479 w_k1_ADM_s100
                                 + 0.675808 w_k1_AXP_s100
                                 + 0.747409 w_k1_DIS_s100
                                 + 0.738178 w_k1_FDX_s100
                                 + 0.687415 w_k1_IBM_s100
                                 + 0.818858 w_k1_INTC_s100
                                 + 0.782277 w_k1_KO_s100
                                 + 0.599029 w_k1_MAT_s100
                                 + 0.917736 w_k1_MSFT_s100
                                 + 0.787416 w_k1_PG_s100 >= 1
 z_k1_s1007_constraint:          var_1 + z_k1_s1007 + 1.0719 w_k1_ADM_s100
                                 + 0.982849 w_k1_AXP_s100
                                 + 1.01617 w_k1_DIS_s100
                                 + 0.992027 w_k1_FDX_s100
                                 + 0.881227 w_k1_IBM_s100
                                 + 1.03418 w_k1_INTC_s100
                                 + 0.930179 w_k1_KO_s100
                                 + 0.775458 w_k1_MAT_s100
                                 + 0.906361 w_k1_MSFT_s100
                                 + 0.922983 w_k1_PG_s100 >= 1
 z_k1_s1010_constraint:          var_1 + z_k1_s1010 + 0.926427 w_k1_ADM_s101
                                 + 0.889176 w_k1_AXP_s101
                                 + 0.884043 w_k1_DIS_s101
                                 + 0.894403 w_k1_FDX_s101
                                 + 1.1641 w_k1_IBM_s101
                                 + 0.912186 w_k1_INTC_s101
                                 + 0.889664 w_k1_KO_s101
                                 + 0.69252 w_k1_MAT_s101
                                 + 0.841958 w_k1_MSFT_s101
                                 + 0.882614 w_k1_PG_s101 >= 1
 z_k1_s1011_constraint:          var_1 + z_k1_s1011 + 1.07122 w_k1_ADM_s101
                                 + 0.818482 w_k1_AXP_s101
                                 + 0.987587 w_k1_DIS_s101
                                 + 1.01742 w_k1_FDX_s101
                                 + 0.770742 w_k1_IBM_s101
                                 + 0.865011 w_k1_INTC_s101
                                 + 1.05365 w_k1_KO_s101
                                 + 0.668731 w_k1_MAT_s101
                                 + 1.06627 w_k1_MSFT_s101
                                 + 0.980177 w_k1_PG_s101 >= 1
 z_k1_s1012_constraint:          var_1 + z_k1_s1012 + 1.3074 w_k1_ADM_s101
                                 + 1.39141 w_k1_AXP_s101
                                 + 1.22946 w_k1_DIS_s101
                                 + 1.22452 w_k1_FDX_s101
                                 + 1.20641 w_k1_IBM_s101
                                 + 1.03217 w_k1_INTC_s101
                                 + 1.11186 w_k1_KO_s101 + 1.32566 w_k1_MAT_s101
                                 + 1.18503 w_k1_MSFT_s101
                                 + 1.15746 w_k1_PG_s101 >= 1
 z_k1_s1013_constraint:          var_1 + z_k1_s1013 + 0.954421 w_k1_ADM_s101
                                 + 0.807526 w_k1_AXP_s101
                                 + 0.697462 w_k1_DIS_s101
                                 + 0.896779 w_k1_FDX_s101
                                 + 0.679904 w_k1_IBM_s101
                                 + 0.810488 w_k1_INTC_s101
                                 + 0.772895 w_k1_KO_s101
                                 + 0.814817 w_k1_MAT_s101
                                 + 0.835499 w_k1_MSFT_s101
                                 + 0.792961 w_k1_PG_s101 >= 1
 z_k1_s1014_constraint:          var_1 + z_k1_s1014 + 1.00982 w_k1_ADM_s101
                                 + 0.717328 w_k1_AXP_s101
                                 + 0.875887 w_k1_DIS_s101
                                 + 0.850435 w_k1_FDX_s101
                                 + 0.859991 w_k1_IBM_s101
                                 + 0.913873 w_k1_INTC_s101
                                 + 0.928489 w_k1_KO_s101
                                 + 0.672033 w_k1_MAT_s101
                                 + 0.966696 w_k1_MSFT_s101
                                 + 0.956252 w_k1_PG_s101 >= 1
 z_k1_s1015_constraint:          var_1 + z_k1_s1015 + 0.894023 w_k1_ADM_s101
                                 + 0.654419 w_k1_AXP_s101
                                 + 0.678433 w_k1_DIS_s101
                                 + 0.688709 w_k1_FDX_s101
                                 + 0.885203 w_k1_IBM_s101
                                 + 0.679469 w_k1_INTC_s101
                                 + 0.751556 w_k1_KO_s101
                                 + 0.72996 w_k1_MAT_s101
                                 + 0.734848 w_k1_MSFT_s101
                                 + 0.80259 w_k1_PG_s101 >= 1
 z_k1_s1016_constraint:          var_1 + z_k1_s1016 + 1.29399 w_k1_ADM_s101
                                 + 1.04204 w_k1_AXP_s101
                                 + 1.39357 w_k1_DIS_s101
                                 + 1.14992 w_k1_FDX_s101
                                 + 1.54721 w_k1_IBM_s101
                                 + 0.98882 w_k1_INTC_s101
                                 + 1.19569 w_k1_KO_s101 + 1.18423 w_k1_MAT_s101
                                 + 1.16594 w_k1_MSFT_s101
                                 + 1.01758 w_k1_PG_s101 >= 1
 z_k1_s1017_constraint:          var_1 + z_k1_s1017 + 1.34263 w_k1_ADM_s101
                                 + 1.06427 w_k1_AXP_s101
                                 + 1.57866 w_k1_DIS_s101
                                 + 1.25633 w_k1_FDX_s101
                                 + 1.06102 w_k1_IBM_s101
                                 + 1.25395 w_k1_INTC_s101 + 1.265 w_k1_KO_s101
                                 + 0.791872 w_k1_MAT_s101
                                 + 1.29114 w_k1_MSFT_s101
                                 + 1.16226 w_k1_PG_s101 >= 1
 z_k1_s1020_constraint:          var_1 + z_k1_s1020 + 0.727802 w_k1_ADM_s102
                                 + 0.750807 w_k1_AXP_s102
                                 + 0.627868 w_k1_DIS_s102
                                 + 0.746339 w_k1_FDX_s102
                                 + 0.741304 w_k1_IBM_s102
                                 + 0.790834 w_k1_INTC_s102
                                 + 0.604814 w_k1_KO_s102
                                 + 0.522023 w_k1_MAT_s102
                                 + 0.682698 w_k1_MSFT_s102
                                 + 0.699667 w_k1_PG_s102 >= 1
 z_k1_s1021_constraint:          var_1 + z_k1_s1021 + 1.17365 w_k1_ADM_s102
                                 + 1.11963 w_k1_AXP_s102
                                 + 0.811976 w_k1_DIS_s102
                                 + 1.0236 w_k1_FDX_s102 + 1.01414 w_k1_IBM_s102
                                 + 1.22376 w_k1_INTC_s102
                                 + 0.965308 w_k1_KO_s102
                                 + 0.804327 w_k1_MAT_s102
                                 + 0.948732 w_k1_MSFT_s102
                                 + 0.97813 w_k1_PG_s102 >= 1
 z_k1_s1022_constraint:          var_1 + z_k1_s1022 + 1.37782 w_k1_ADM_s102
                                 + 1.53674 w_k1_AXP_s102
                                 + 1.49345 w_k1_DIS_s102
                                 + 1.36521 w_k1_FDX_s102
                                 + 1.25626 w_k1_IBM_s102
                                 + 1.2603 w_k1_INTC_s102 + 1.04467 w_k1_KO_s102
                                 + 0.843606 w_k1_MAT_s102
                                 + 1.26318 w_k1_MSFT_s102
                                 + 1.07511 w_k1_PG_s102 >= 1
 z_k1_s1023_constraint:          var_1 + z_k1_s1023 + 0.75014 w_k1_ADM_s102
                                 + 0.881013 w_k1_AXP_s102
                                 + 0.955513 w_k1_DIS_s102
                                 + 0.968689 w_k1_FDX_s102
                                 + 0.802221 w_k1_IBM_s102
                                 + 1.09899 w_k1_INTC_s102
                                 + 0.924415 w_k1_KO_s102
                                 + 0.710508 w_k1_MAT_s102
                                 + 1.1078 w_k1_MSFT_s102
                                 + 0.923911 w_k1_PG_s102 >= 1
 z_k1_s1024_constraint:          var_1 + z_k1_s1024 + 1.24483 w_k1_ADM_s102
                                 + 1.14558 w_k1_AXP_s102
                                 + 1.18154 w_k1_DIS_s102
                                 + 1.17259 w_k1_FDX_s102
                                 + 0.809094 w_k1_IBM_s102
                                 + 1.23133 w_k1_INTC_s102
                                 + 1.03547 w_k1_KO_s102
                                 + 0.692823 w_k1_MAT_s102
                                 + 1.11416 w_k1_MSFT_s102
                                 + 1.00496 w_k1_PG_s102 >= 1
 z_k1_s1025_constraint:          var_1 + z_k1_s1025 + 0.898116 w_k1_ADM_s102
                                 + 0.902632 w_k1_AXP_s102
                                 + 0.865921 w_k1_DIS_s102
                                 + 1.04095 w_k1_FDX_s102
                                 + 0.689217 w_k1_IBM_s102
                                 + 1.03632 w_k1_INTC_s102
                                 + 0.814688 w_k1_KO_s102 + 0.4713 w_k1_MAT_s102
                                 + 0.980714 w_k1_MSFT_s102
                                 + 0.930406 w_k1_PG_s102 >= 1
 z_k1_s1026_constraint:          var_1 + z_k1_s1026 + 0.802945 w_k1_ADM_s102
                                 + 0.767278 w_k1_AXP_s102
                                 + 0.688792 w_k1_DIS_s102
                                 + 0.797216 w_k1_FDX_s102
                                 + 0.815885 w_k1_IBM_s102
                                 + 0.881755 w_k1_INTC_s102
                                 + 0.750729 w_k1_KO_s102
                                 + 0.455036 w_k1_MAT_s102
                                 + 0.830092 w_k1_MSFT_s102
                                 + 0.824775 w_k1_PG_s102 >= 1
 z_k1_s1027_constraint:          var_1 + z_k1_s1027 + 1.11951 w_k1_ADM_s102
                                 + 1.14343 w_k1_AXP_s102
                                 + 0.880416 w_k1_DIS_s102
                                 + 0.921438 w_k1_FDX_s102
                                 + 1.40989 w_k1_IBM_s102
                                 + 1.05778 w_k1_INTC_s102
                                 + 1.13128 w_k1_KO_s102 + 1.10424 w_k1_MAT_s102
                                 + 0.988724 w_k1_MSFT_s102
                                 + 1.06461 w_k1_PG_s102 >= 1
 z_k1_s1030_constraint:          var_1 + z_k1_s1030 + 1.30146 w_k1_ADM_s103
                                 + 1.82451 w_k1_AXP_s103
                                 + 1.60378 w_k1_DIS_s103
                                 + 1.38382 w_k1_FDX_s103
                                 + 1.06301 w_k1_IBM_s103
                                 + 1.24228 w_k1_INTC_s103
                                 + 1.03866 w_k1_KO_s103 + 1.38257 w_k1_MAT_s103
                                 + 1.25331 w_k1_MSFT_s103
                                 + 1.05171 w_k1_PG_s103 >= 1
 z_k1_s1031_constraint:          var_1 + z_k1_s1031 + 0.91268 w_k1_ADM_s103
                                 + 0.720834 w_k1_AXP_s103
                                 + 0.557389 w_k1_DIS_s103
                                 + 0.804179 w_k1_FDX_s103
                                 + 0.676138 w_k1_IBM_s103
                                 + 0.674159 w_k1_INTC_s103
                                 + 0.676103 w_k1_KO_s103
                                 + 0.699501 w_k1_MAT_s103
                                 + 0.649361 w_k1_MSFT_s103
                                 + 0.815139 w_k1_PG_s103 >= 1
 z_k1_s1032_constraint:          var_1 + z_k1_s1032 + 0.945154 w_k1_ADM_s103
                                 + 0.903197 w_k1_AXP_s103
                                 + 1.00987 w_k1_DIS_s103
                                 + 1.05117 w_k1_FDX_s103
                                 + 0.940857 w_k1_IBM_s103
                                 + 0.942794 w_k1_INTC_s103
                                 + 0.941342 w_k1_KO_s103
                                 + 0.871736 w_k1_MAT_s103
                                 + 0.997292 w_k1_MSFT_s103
                                 + 0.96252 w_k1_PG_s103 >= 1
 z_k1_s1033_constraint:          var_1 + z_k1_s1033 + 1.20883 w_k1_ADM_s103
                                 + 1.15921 w_k1_AXP_s103
                                 + 1.23358 w_k1_DIS_s103
                                 + 1.28326 w_k1_FDX_s103
                                 + 0.827038 w_k1_IBM_s103
                                 + 1.08817 w_k1_INTC_s103
                                 + 0.908391 w_k1_KO_s103
                                 + 0.602035 w_k1_MAT_s103
                                 + 0.991949 w_k1_MSFT_s103
                                 + 0.94825 w_k1_PG_s103 >= 1
 z_k1_s1034_constraint:          var_1 + z_k1_s1034 + 1.23221 w_k1_ADM_s103
                                 + 1.52701 w_k1_AXP_s103
                                 + 1.25132 w_k1_DIS_s103
                                 + 1.10187 w_k1_FDX_s103
                                 + 1.52318 w_k1_IBM_s103
                                 + 1.1693 w_k1_INTC_s103 + 1.26853 w_k1_KO_s103
                                 + 1.27884 w_k1_MAT_s103
                                 + 0.991174 w_k1_MSFT_s103
                                 + 1.15512 w_k1_PG_s103 >= 1
 z_k1_s1035_constraint:          var_1 + z_k1_s1035 + 1.15228 w_k1_ADM_s103
                                 + 0.845823 w_k1_AXP_s103
                                 + 0.887253 w_k1_DIS_s103
                                 + 0.87855 w_k1_FDX_s103
                                 + 1.23867 w_k1_IBM_s103
                                 + 0.830727 w_k1_INTC_s103
                                 + 0.971811 w_k1_KO_s103
                                 + 0.951064 w_k1_MAT_s103
                                 + 0.82662 w_k1_MSFT_s103
                                 + 0.946708 w_k1_PG_s103 >= 1
 z_k1_s1036_constraint:          var_1 + z_k1_s1036 + 1.12439 w_k1_ADM_s103
                                 + 0.928713 w_k1_AXP_s103
                                 + 0.787464 w_k1_DIS_s103
                                 + 1.01143 w_k1_FDX_s103
                                 + 0.568131 w_k1_IBM_s103
                                 + 0.858903 w_k1_INTC_s103
                                 + 0.625169 w_k1_KO_s103
                                 + 0.562933 w_k1_MAT_s103
                                 + 0.842056 w_k1_MSFT_s103
                                 + 0.782716 w_k1_PG_s103 >= 1
 z_k1_s1037_constraint:          var_1 + z_k1_s1037 + 0.93155 w_k1_ADM_s103
                                 + 0.892474 w_k1_AXP_s103
                                 + 0.631584 w_k1_DIS_s103
                                 + 0.737949 w_k1_FDX_s103
                                 + 1.09155 w_k1_IBM_s103
                                 + 0.720397 w_k1_INTC_s103
                                 + 0.712208 w_k1_KO_s103
                                 + 0.965594 w_k1_MAT_s103
                                 + 0.660691 w_k1_MSFT_s103
                                 + 0.795014 w_k1_PG_s103 >= 1
 z_k1_s1040_constraint:          var_1 + z_k1_s1040 + 0.980271 w_k1_ADM_s104
                                 + 1.16653 w_k1_AXP_s104 + 0.9487 w_k1_DIS_s104
                                 + 0.899625 w_k1_FDX_s104
                                 + 1.30293 w_k1_IBM_s104
                                 + 0.77046 w_k1_INTC_s104
                                 + 0.962982 w_k1_KO_s104
                                 + 1.50208 w_k1_MAT_s104
                                 + 0.932946 w_k1_MSFT_s104
                                 + 1.02493 w_k1_PG_s104 >= 1
 z_k1_s1041_constraint:          var_1 + z_k1_s1041 + 1.30439 w_k1_ADM_s104
                                 + 1.8327 w_k1_AXP_s104 + 1.50933 w_k1_DIS_s104
                                 + 1.40091 w_k1_FDX_s104
                                 + 1.02259 w_k1_IBM_s104
                                 + 1.20787 w_k1_INTC_s104
                                 + 1.10956 w_k1_KO_s104 + 1.66005 w_k1_MAT_s104
                                 + 1.27759 w_k1_MSFT_s104
                                 + 1.19448 w_k1_PG_s104 >= 1
 z_k1_s1042_constraint:          var_1 + z_k1_s1042 + 0.990199 w_k1_ADM_s104
                                 + 0.818703 w_k1_AXP_s104
                                 + 0.949232 w_k1_DIS_s104
                                 + 0.993575 w_k1_FDX_s104
                                 + 0.698172 w_k1_IBM_s104
                                 + 0.905869 w_k1_INTC_s104
                                 + 0.730567 w_k1_KO_s104
                                 + 0.687836 w_k1_MAT_s104
                                 + 0.95759 w_k1_MSFT_s104
                                 + 0.971793 w_k1_PG_s104 >= 1
 z_k1_s1043_constraint:          var_1 + z_k1_s1043 + 1.01917 w_k1_ADM_s104
                                 + 0.852304 w_k1_AXP_s104
                                 + 0.84225 w_k1_DIS_s104
                                 + 0.865202 w_k1_FDX_s104
                                 + 0.855493 w_k1_IBM_s104
                                 + 0.698799 w_k1_INTC_s104
                                 + 0.7688 w_k1_KO_s104 + 0.888554 w_k1_MAT_s104
                                 + 0.895477 w_k1_MSFT_s104
                                 + 0.904404 w_k1_PG_s104 >= 1
 z_k1_s1044_constraint:          var_1 + z_k1_s1044 + 0.793199 w_k1_ADM_s104
                                 + 0.826605 w_k1_AXP_s104
                                 + 0.626269 w_k1_DIS_s104
                                 + 0.724719 w_k1_FDX_s104
                                 + 0.912092 w_k1_IBM_s104
                                 + 0.78335 w_k1_INTC_s104
                                 + 0.737168 w_k1_KO_s104
                                 + 0.846762 w_k1_MAT_s104
                                 + 0.717073 w_k1_MSFT_s104
                                 + 0.905274 w_k1_PG_s104 >= 1
 z_k1_s1045_constraint:          var_1 + z_k1_s1045 + 1.52193 w_k1_ADM_s104
                                 + 1.21354 w_k1_AXP_s104
                                 + 1.68047 w_k1_DIS_s104
                                 + 1.36726 w_k1_FDX_s104
                                 + 0.961818 w_k1_IBM_s104
                                 + 1.15573 w_k1_INTC_s104
                                 + 1.08375 w_k1_KO_s104
                                 + 0.937908 w_k1_MAT_s104
                                 + 1.2404 w_k1_MSFT_s104 + 1.12851 w_k1_PG_s104
                                 >= 1
 z_k1_s1046_constraint:          var_1 + z_k1_s1046 + 0.939106 w_k1_ADM_s104
                                 + 1.18104 w_k1_AXP_s104
                                 + 1.01956 w_k1_DIS_s104 + 1.0726 w_k1_FDX_s104
                                 + 0.854063 w_k1_IBM_s104
                                 + 1.0327 w_k1_INTC_s104
                                 + 0.898472 w_k1_KO_s104
                                 + 1.26172 w_k1_MAT_s104
                                 + 1.04288 w_k1_MSFT_s104
                                 + 0.929476 w_k1_PG_s104 >= 1
 z_k1_s1047_constraint:          var_1 + z_k1_s1047 + 1.05077 w_k1_ADM_s104
                                 + 1.10604 w_k1_AXP_s104
                                 + 1.37291 w_k1_DIS_s104
                                 + 1.11834 w_k1_FDX_s104 + 1.317 w_k1_IBM_s104
                                 + 1.21993 w_k1_INTC_s104 + 1.2032 w_k1_KO_s104
                                 + 1.2485 w_k1_MAT_s104
                                 + 1.22759 w_k1_MSFT_s104
                                 + 0.992513 w_k1_PG_s104 >= 1
 z_k1_s1050_constraint:          var_1 + z_k1_s1050 + 0.77613 w_k1_ADM_s105
                                 + 0.830049 w_k1_AXP_s105
                                 + 0.787008 w_k1_DIS_s105
                                 + 0.802124 w_k1_FDX_s105
                                 + 1.23132 w_k1_IBM_s105
                                 + 0.788426 w_k1_INTC_s105
                                 + 1.08024 w_k1_KO_s105
                                 + 0.988444 w_k1_MAT_s105
                                 + 0.92082 w_k1_MSFT_s105
                                 + 1.12098 w_k1_PG_s105 >= 1
 z_k1_s1051_constraint:          var_1 + z_k1_s1051 + 1.299 w_k1_ADM_s105
                                 + 1.35389 w_k1_AXP_s105
                                 + 1.16999 w_k1_DIS_s105
                                 + 1.09467 w_k1_FDX_s105
                                 + 1.09715 w_k1_IBM_s105
                                 + 0.972229 w_k1_INTC_s105
                                 + 1.14624 w_k1_KO_s105 + 1.39674 w_k1_MAT_s105
                                 + 1.1442 w_k1_MSFT_s105 + 1.13609 w_k1_PG_s105
                                 >= 1
 z_k1_s1052_constraint:          var_1 + z_k1_s1052 + 1.10318 w_k1_ADM_s105
                                 + 1.18683 w_k1_AXP_s105
                                 + 1.29122 w_k1_DIS_s105
                                 + 1.22213 w_k1_FDX_s105
                                 + 1.57723 w_k1_IBM_s105
                                 + 1.16161 w_k1_INTC_s105
                                 + 1.50639 w_k1_KO_s105 + 1.89295 w_k1_MAT_s105
                                 + 1.37687 w_k1_MSFT_s105
                                 + 1.24968 w_k1_PG_s105 >= 1
 z_k1_s1053_constraint:          var_1 + z_k1_s1053 + 1.16129 w_k1_ADM_s105
                                 + 1.2932 w_k1_AXP_s105 + 1.22154 w_k1_DIS_s105
                                 + 1.31708 w_k1_FDX_s105
                                 + 0.886302 w_k1_IBM_s105
                                 + 1.18707 w_k1_INTC_s105
                                 + 1.19579 w_k1_KO_s105 + 1.10861 w_k1_MAT_s105
                                 + 1.38308 w_k1_MSFT_s105
                                 + 1.07574 w_k1_PG_s105 >= 1
 z_k1_s1054_constraint:          var_1 + z_k1_s1054 + 0.688572 w_k1_ADM_s105
                                 + 1.01916 w_k1_AXP_s105
                                 + 0.74321 w_k1_DIS_s105
                                 + 0.870534 w_k1_FDX_s105
                                 + 0.762448 w_k1_IBM_s105
                                 + 0.915305 w_k1_INTC_s105
                                 + 0.772433 w_k1_KO_s105
                                 + 1.32273 w_k1_MAT_s105
                                 + 0.982059 w_k1_MSFT_s105
                                 + 0.924971 w_k1_PG_s105 >= 1
 z_k1_s1055_constraint:          var_1 + z_k1_s1055 + 0.830767 w_k1_ADM_s105
                                 + 0.701697 w_k1_AXP_s105
                                 + 0.83704 w_k1_DIS_s105
                                 + 0.938741 w_k1_FDX_s105
                                 + 0.835356 w_k1_IBM_s105
                                 + 1.18233 w_k1_INTC_s105
                                 + 1.01815 w_k1_KO_s105
                                 + 0.630159 w_k1_MAT_s105
                                 + 0.961136 w_k1_MSFT_s105
                                 + 1.04112 w_k1_PG_s105 >= 1
 z_k1_s1056_constraint:          var_1 + z_k1_s1056 + 0.790964 w_k1_ADM_s105
                                 + 0.86489 w_k1_AXP_s105
                                 + 0.676766 w_k1_DIS_s105
                                 + 0.78895 w_k1_FDX_s105
                                 + 0.621694 w_k1_IBM_s105
                                 + 0.768927 w_k1_INTC_s105
                                 + 0.700709 w_k1_KO_s105
                                 + 0.812268 w_k1_MAT_s105
                                 + 0.798168 w_k1_MSFT_s105
                                 + 0.950069 w_k1_PG_s105 >= 1
 z_k1_s1057_constraint:          var_1 + z_k1_s1057 + 0.636659 w_k1_ADM_s105
                                 + 0.632295 w_k1_AXP_s105
                                 + 0.586873 w_k1_DIS_s105
                                 + 0.708889 w_k1_FDX_s105
                                 + 0.897421 w_k1_IBM_s105
                                 + 0.762382 w_k1_INTC_s105
                                 + 0.784557 w_k1_KO_s105
                                 + 0.834125 w_k1_MAT_s105
                                 + 0.837327 w_k1_MSFT_s105
                                 + 0.846598 w_k1_PG_s105 >= 1
 z_k1_s1060_constraint:          var_1 + z_k1_s1060 + 0.950349 w_k1_ADM_s106
                                 + 0.914944 w_k1_AXP_s106
                                 + 0.679867 w_k1_DIS_s106
                                 + 0.813241 w_k1_FDX_s106
                                 + 0.960729 w_k1_IBM_s106
                                 + 0.704822 w_k1_INTC_s106
                                 + 0.7068 w_k1_KO_s106 + 1.13615 w_k1_MAT_s106
                                 + 0.747626 w_k1_MSFT_s106
                                 + 0.896889 w_k1_PG_s106 >= 1
 z_k1_s1061_constraint:          var_1 + z_k1_s1061 + 1.12429 w_k1_ADM_s106
                                 + 1.64968 w_k1_AXP_s106
                                 + 1.85485 w_k1_DIS_s106
                                 + 1.56936 w_k1_FDX_s106
                                 + 1.02283 w_k1_IBM_s106
                                 + 1.15997 w_k1_INTC_s106
                                 + 1.11636 w_k1_KO_s106 + 1.27563 w_k1_MAT_s106
                                 + 1.18119 w_k1_MSFT_s106
                                 + 1.05262 w_k1_PG_s106 >= 1
 z_k1_s1062_constraint:          var_1 + z_k1_s1062 + 1.12504 w_k1_ADM_s106
                                 + 1.39197 w_k1_AXP_s106
                                 + 1.36786 w_k1_DIS_s106
                                 + 1.11004 w_k1_FDX_s106
                                 + 1.48688 w_k1_IBM_s106
                                 + 0.971823 w_k1_INTC_s106
                                 + 1.40432 w_k1_KO_s106 + 1.88371 w_k1_MAT_s106
                                 + 1.0085 w_k1_MSFT_s106 + 1.05523 w_k1_PG_s106
                                 >= 1
 z_k1_s1063_constraint:          var_1 + z_k1_s1063 + 0.839568 w_k1_ADM_s106
                                 + 0.812323 w_k1_AXP_s106
                                 + 1.29702 w_k1_DIS_s106
                                 + 1.04951 w_k1_FDX_s106
                                 + 0.84795 w_k1_IBM_s106
                                 + 0.889068 w_k1_INTC_s106
                                 + 0.897851 w_k1_KO_s106
                                 + 0.81509 w_k1_MAT_s106
                                 + 1.03795 w_k1_MSFT_s106
                                 + 0.952495 w_k1_PG_s106 >= 1
 z_k1_s1064_constraint:          var_1 + z_k1_s1064 + 0.900847 w_k1_ADM_s106
                                 + 0.791771 w_k1_AXP_s106
                                 + 0.828948 w_k1_DIS_s106
                                 + 0.982204 w_k1_FDX_s106
                                 + 0.66015 w_k1_IBM_s106
                                 + 0.916433 w_k1_INTC_s106
                                 + 0.6954 w_k1_KO_s106 + 0.655703 w_k1_MAT_s106
                                 + 0.991179 w_k1_MSFT_s106
                                 + 0.913591 w_k1_PG_s106 >= 1
 z_k1_s1065_constraint:          var_1 + z_k1_s1065 + 0.97915 w_k1_ADM_s106
                                 + 1.10864 w_k1_AXP_s106
                                 + 1.07656 w_k1_DIS_s106
                                 + 1.15092 w_k1_FDX_s106
                                 + 0.919861 w_k1_IBM_s106
                                 + 0.941352 w_k1_INTC_s106
                                 + 0.801858 w_k1_KO_s106
                                 + 0.808191 w_k1_MAT_s106
                                 + 1.00896 w_k1_MSFT_s106
                                 + 0.934474 w_k1_PG_s106 >= 1
 z_k1_s1066_constraint:          var_1 + z_k1_s1066 + 0.894262 w_k1_ADM_s106
                                 + 0.750097 w_k1_AXP_s106
                                 + 0.945636 w_k1_DIS_s106
                                 + 0.838886 w_k1_FDX_s106
                                 + 1.14662 w_k1_IBM_s106
                                 + 0.76992 w_k1_INTC_s106
                                 + 0.933616 w_k1_KO_s106
                                 + 0.846098 w_k1_MAT_s106
                                 + 0.977417 w_k1_MSFT_s106
                                 + 0.884543 w_k1_PG_s106 >= 1
 z_k1_s1067_constraint:          var_1 + z_k1_s1067 + 1.3784 w_k1_ADM_s106
                                 + 1.4487 w_k1_AXP_s106 + 1.20088 w_k1_DIS_s106
                                 + 1.258 w_k1_FDX_s106 + 0.781452 w_k1_IBM_s106
                                 + 1.06427 w_k1_INTC_s106
                                 + 0.851186 w_k1_KO_s106
                                 + 0.94455 w_k1_MAT_s106
                                 + 1.10472 w_k1_MSFT_s106
                                 + 0.936957 w_k1_PG_s106 >= 1
 z_k1_s1070_constraint:          var_1 + z_k1_s1070 + 0.991661 w_k1_ADM_s107
                                 + 0.880173 w_k1_AXP_s107
                                 + 0.897776 w_k1_DIS_s107
                                 + 0.954344 w_k1_FDX_s107
                                 + 1.35258 w_k1_IBM_s107
                                 + 0.955957 w_k1_INTC_s107
                                 + 1.05248 w_k1_KO_s107 + 1.20035 w_k1_MAT_s107
                                 + 0.972423 w_k1_MSFT_s107
                                 + 0.981877 w_k1_PG_s107 >= 1
 z_k1_s1071_constraint:          var_1 + z_k1_s1071 + 1.35181 w_k1_ADM_s107
                                 + 1.16163 w_k1_AXP_s107
                                 + 1.25674 w_k1_DIS_s107
                                 + 1.30046 w_k1_FDX_s107
                                 + 0.747331 w_k1_IBM_s107
                                 + 1.13229 w_k1_INTC_s107
                                 + 0.969547 w_k1_KO_s107
                                 + 0.658506 w_k1_MAT_s107
                                 + 1.24078 w_k1_MSFT_s107 + 1.0987 w_k1_PG_s107
                                 >= 1
 z_k1_s1072_constraint:          var_1 + z_k1_s1072 + 0.934958 w_k1_ADM_s107
                                 + 1.00533 w_k1_AXP_s107
                                 + 0.878669 w_k1_DIS_s107
                                 + 0.926664 w_k1_FDX_s107
                                 + 0.837718 w_k1_IBM_s107
                                 + 1.13914 w_k1_INTC_s107
                                 + 0.855706 w_k1_KO_s107
                                 + 0.988392 w_k1_MAT_s107
                                 + 1.06511 w_k1_MSFT_s107
                                 + 0.892859 w_k1_PG_s107 >= 1
 z_k1_s1073_constraint:          var_1 + z_k1_s1073 + 1.7032 w_k1_ADM_s107
                                 + 1.72862 w_k1_AXP_s107
                                 + 1.96769 w_k1_DIS_s107
                                 + 1.68044 w_k1_FDX_s107
                                 + 1.37092 w_k1_IBM_s107
                                 + 1.59908 w_k1_INTC_s107
                                 + 1.67333 w_k1_KO_s107 + 1.48769 w_k1_MAT_s107
                                 + 1.61126 w_k1_MSFT_s107
                                 + 1.42359 w_k1_PG_s107 >= 1
 z_k1_s1074_constraint:          var_1 + z_k1_s1074 + 1.15451 w_k1_ADM_s107
                                 + 1.32883 w_k1_AXP_s107
                                 + 0.896039 w_k1_DIS_s107
                                 + 1.08406 w_k1_FDX_s107
                                 + 0.968494 w_k1_IBM_s107
                                 + 1.32014 w_k1_INTC_s107
                                 + 1.00716 w_k1_KO_s107
                                 + 0.885508 w_k1_MAT_s107
                                 + 1.04619 w_k1_MSFT_s107
                                 + 0.947072 w_k1_PG_s107 >= 1
 z_k1_s1075_constraint:          var_1 + z_k1_s1075 + 0.820232 w_k1_ADM_s107
                                 + 1.01012 w_k1_AXP_s107
                                 + 0.64593 w_k1_DIS_s107
                                 + 0.78485 w_k1_FDX_s107
                                 + 0.939865 w_k1_IBM_s107
                                 + 0.869129 w_k1_INTC_s107
                                 + 0.749554 w_k1_KO_s107
                                 + 0.849708 w_k1_MAT_s107
                                 + 0.726322 w_k1_MSFT_s107
                                 + 0.842012 w_k1_PG_s107 >= 1
 z_k1_s1076_constraint:          var_1 + z_k1_s1076 + 1.25108 w_k1_ADM_s107
                                 + 1.54485 w_k1_AXP_s107
                                 + 1.29426 w_k1_DIS_s107
                                 + 1.29841 w_k1_FDX_s107
                                 + 1.38872 w_k1_IBM_s107
                                 + 1.52356 w_k1_INTC_s107 + 1.3824 w_k1_KO_s107
                                 + 1.33824 w_k1_MAT_s107
                                 + 1.33984 w_k1_MSFT_s107
                                 + 1.11948 w_k1_PG_s107 >= 1
 z_k1_s1077_constraint:          var_1 + z_k1_s1077 + 1.4151 w_k1_ADM_s107
                                 + 1.20262 w_k1_AXP_s107
                                 + 1.26272 w_k1_DIS_s107
                                 + 1.22675 w_k1_FDX_s107
                                 + 1.22017 w_k1_IBM_s107
                                 + 1.23452 w_k1_INTC_s107
                                 + 1.23411 w_k1_KO_s107
                                 + 0.913347 w_k1_MAT_s107
                                 + 1.19656 w_k1_MSFT_s107
                                 + 1.04211 w_k1_PG_s107 >= 1
 z_k1_s1100_constraint:          var_1 + z_k1_s1100 + 0.914155 w_k1_ADM_s110
                                 + 0.630956 w_k1_AXP_s110
                                 + 0.571803 w_k1_DIS_s110
                                 + 0.771591 w_k1_FDX_s110
                                 + 0.66821 w_k1_IBM_s110
                                 + 0.833471 w_k1_INTC_s110
                                 + 0.831322 w_k1_KO_s110
                                 + 0.628323 w_k1_MAT_s110
                                 + 0.766548 w_k1_MSFT_s110
                                 + 0.8736 w_k1_PG_s110 >= 1
 z_k1_s1101_constraint:          var_1 + z_k1_s1101 + 0.908215 w_k1_ADM_s110
                                 + 0.599908 w_k1_AXP_s110
                                 + 0.543253 w_k1_DIS_s110
                                 + 0.652757 w_k1_FDX_s110
                                 + 0.90161 w_k1_IBM_s110
                                 + 0.656784 w_k1_INTC_s110
                                 + 0.897212 w_k1_KO_s110
                                 + 0.622714 w_k1_MAT_s110
                                 + 0.565195 w_k1_MSFT_s110
                                 + 0.739517 w_k1_PG_s110 >= 1
 z_k1_s1102_constraint:          var_1 + z_k1_s1102 + 1.09884 w_k1_ADM_s110
                                 + 1.0207 w_k1_AXP_s110
                                 + 0.730239 w_k1_DIS_s110
                                 + 0.872186 w_k1_FDX_s110 + 1.124 w_k1_IBM_s110
                                 + 0.812314 w_k1_INTC_s110
                                 + 0.980751 w_k1_KO_s110
                                 + 1.16001 w_k1_MAT_s110
                                 + 0.65815 w_k1_MSFT_s110
                                 + 0.881705 w_k1_PG_s110 >= 1
 z_k1_s1103_constraint:          var_1 + z_k1_s1103 + 0.992462 w_k1_ADM_s110
                                 + 0.753217 w_k1_AXP_s110
                                 + 0.831719 w_k1_DIS_s110
                                 + 0.931379 w_k1_FDX_s110
                                 + 0.887449 w_k1_IBM_s110
                                 + 0.924139 w_k1_INTC_s110
                                 + 1.17916 w_k1_KO_s110
                                 + 0.804122 w_k1_MAT_s110
                                 + 0.926679 w_k1_MSFT_s110
                                 + 0.967392 w_k1_PG_s110 >= 1
 z_k1_s1104_constraint:          var_1 + z_k1_s1104 + 1.02266 w_k1_ADM_s110
                                 + 1.30748 w_k1_AXP_s110
                                 + 0.761914 w_k1_DIS_s110
                                 + 0.989863 w_k1_FDX_s110
                                 + 0.992965 w_k1_IBM_s110
                                 + 1.08802 w_k1_INTC_s110
                                 + 0.906396 w_k1_KO_s110
                                 + 1.16026 w_k1_MAT_s110
                                 + 0.901111 w_k1_MSFT_s110
                                 + 0.903534 w_k1_PG_s110 >= 1
 z_k1_s1105_constraint:          var_1 + z_k1_s1105 + 1.5697 w_k1_ADM_s110
                                 + 1.2651 w_k1_AXP_s110 + 1.62639 w_k1_DIS_s110
                                 + 1.45628 w_k1_FDX_s110
                                 + 1.25477 w_k1_IBM_s110
                                 + 1.12166 w_k1_INTC_s110
                                 + 1.57933 w_k1_KO_s110 + 1.19482 w_k1_MAT_s110
                                 + 1.33515 w_k1_MSFT_s110
                                 + 1.22084 w_k1_PG_s110 >= 1
 z_k1_s1106_constraint:          var_1 + z_k1_s1106 + 1.10673 w_k1_ADM_s110
                                 + 1.24978 w_k1_AXP_s110
                                 + 1.01404 w_k1_DIS_s110
                                 + 1.09083 w_k1_FDX_s110
                                 + 1.25401 w_k1_IBM_s110
                                 + 1.07817 w_k1_INTC_s110
                                 + 1.19604 w_k1_KO_s110 + 1.3875 w_k1_MAT_s110
                                 + 0.957818 w_k1_MSFT_s110
                                 + 1.01422 w_k1_PG_s110 >= 1
 z_k1_s1107_constraint:          var_1 + z_k1_s1107 + 1.13317 w_k1_ADM_s110
                                 + 0.884473 w_k1_AXP_s110
                                 + 0.955619 w_k1_DIS_s110
                                 + 1.12728 w_k1_FDX_s110
                                 + 0.798069 w_k1_IBM_s110
                                 + 1.0357 w_k1_INTC_s110 + 1.01371 w_k1_KO_s110
                                 + 0.557029 w_k1_MAT_s110
                                 + 1.11889 w_k1_MSFT_s110
                                 + 0.990768 w_k1_PG_s110 >= 1
 z_k1_s1110_constraint:          var_1 + z_k1_s1110 + 0.946501 w_k1_ADM_s111
                                 + 1.06152 w_k1_AXP_s111
                                 + 0.96334 w_k1_DIS_s111
                                 + 0.916058 w_k1_FDX_s111
                                 + 0.943888 w_k1_IBM_s111
                                 + 0.929664 w_k1_INTC_s111
                                 + 0.99787 w_k1_KO_s111
                                 + 0.928048 w_k1_MAT_s111
                                 + 0.871017 w_k1_MSFT_s111
                                 + 0.986777 w_k1_PG_s111 >= 1
 z_k1_s1111_constraint:          var_1 + z_k1_s1111 + 1.01062 w_k1_ADM_s111
                                 + 1.02128 w_k1_AXP_s111
                                 + 0.990437 w_k1_DIS_s111
                                 + 1.04605 w_k1_FDX_s111
                                 + 0.695891 w_k1_IBM_s111
                                 + 0.85977 w_k1_INTC_s111
                                 + 0.73833 w_k1_KO_s111
                                 + 0.690329 w_k1_MAT_s111
                                 + 1.00505 w_k1_MSFT_s111 + 1.0551 w_k1_PG_s111
                                 >= 1
 z_k1_s1112_constraint:          var_1 + z_k1_s1112 + 1.13617 w_k1_ADM_s111
                                 + 1.35235 w_k1_AXP_s111
                                 + 1.59714 w_k1_DIS_s111
                                 + 1.25718 w_k1_FDX_s111
                                 + 0.986131 w_k1_IBM_s111
                                 + 1.21159 w_k1_INTC_s111 + 1.1211 w_k1_KO_s111
                                 + 0.720731 w_k1_MAT_s111
                                 + 1.25109 w_k1_MSFT_s111
                                 + 1.05942 w_k1_PG_s111 >= 1
 z_k1_s1113_constraint:          var_1 + z_k1_s1113 + 0.599215 w_k1_ADM_s111
                                 + 1.18538 w_k1_AXP_s111
                                 + 0.942265 w_k1_DIS_s111
                                 + 0.845857 w_k1_FDX_s111
                                 + 1.35268 w_k1_IBM_s111
                                 + 1.09759 w_k1_INTC_s111
                                 + 1.16962 w_k1_KO_s111 + 1.45498 w_k1_MAT_s111
                                 + 0.944507 w_k1_MSFT_s111
                                 + 1.07418 w_k1_PG_s111 >= 1
 z_k1_s1114_constraint:          var_1 + z_k1_s1114 + 0.982627 w_k1_ADM_s111
                                 + 1.71989 w_k1_AXP_s111
                                 + 1.54857 w_k1_DIS_s111
                                 + 1.27332 w_k1_FDX_s111
                                 + 1.25922 w_k1_IBM_s111
                                 + 1.28229 w_k1_INTC_s111
                                 + 1.32478 w_k1_KO_s111 + 1.34688 w_k1_MAT_s111
                                 + 1.12799 w_k1_MSFT_s111
                                 + 1.26326 w_k1_PG_s111 >= 1
 z_k1_s1115_constraint:          var_1 + z_k1_s1115 + 0.719819 w_k1_ADM_s111
                                 + 0.645855 w_k1_AXP_s111
                                 + 0.643586 w_k1_DIS_s111
                                 + 0.705863 w_k1_FDX_s111
                                 + 0.737157 w_k1_IBM_s111
                                 + 0.762965 w_k1_INTC_s111
                                 + 0.71148 w_k1_KO_s111
                                 + 0.613695 w_k1_MAT_s111
                                 + 0.698275 w_k1_MSFT_s111
                                 + 0.805825 w_k1_PG_s111 >= 1
 z_k1_s1116_constraint:          var_1 + z_k1_s1116 + 1.53081 w_k1_ADM_s111
                                 + 1.90772 w_k1_AXP_s111
                                 + 1.66415 w_k1_DIS_s111
                                 + 1.21945 w_k1_FDX_s111 + 1.8242 w_k1_IBM_s111
                                 + 1.29565 w_k1_INTC_s111
                                 + 1.62685 w_k1_KO_s111 + 1.67661 w_k1_MAT_s111
                                 + 1.15375 w_k1_MSFT_s111
                                 + 1.41113 w_k1_PG_s111 >= 1
 z_k1_s1117_constraint:          var_1 + z_k1_s1117 + 0.957742 w_k1_ADM_s111
                                 + 0.871757 w_k1_AXP_s111
                                 + 0.713101 w_k1_DIS_s111
                                 + 0.853025 w_k1_FDX_s111
                                 + 0.602035 w_k1_IBM_s111
                                 + 0.71916 w_k1_INTC_s111
                                 + 0.641559 w_k1_KO_s111
                                 + 0.668507 w_k1_MAT_s111
                                 + 0.706013 w_k1_MSFT_s111
                                 + 0.881825 w_k1_PG_s111 >= 1
 z_k1_s1120_constraint:          var_1 + z_k1_s1120 + 0.877824 w_k1_ADM_s112
                                 + 1.49195 w_k1_AXP_s112 + 1.2769 w_k1_DIS_s112
                                 + 1.19933 w_k1_FDX_s112
                                 + 1.52556 w_k1_IBM_s112
                                 + 1.28648 w_k1_INTC_s112
                                 + 1.29448 w_k1_KO_s112 + 2.18466 w_k1_MAT_s112
                                 + 1.26547 w_k1_MSFT_s112
                                 + 1.10762 w_k1_PG_s112 >= 1
 z_k1_s1121_constraint:          var_1 + z_k1_s1121 + 1.10482 w_k1_ADM_s112
                                 + 0.983118 w_k1_AXP_s112
                                 + 1.00139 w_k1_DIS_s112
                                 + 1.08081 w_k1_FDX_s112
                                 + 0.765845 w_k1_IBM_s112
                                 + 1.08054 w_k1_INTC_s112
                                 + 0.856886 w_k1_KO_s112
                                 + 0.85027 w_k1_MAT_s112
                                 + 0.891064 w_k1_MSFT_s112
                                 + 0.900132 w_k1_PG_s112 >= 1
 z_k1_s1122_constraint:          var_1 + z_k1_s1122 + 0.676251 w_k1_ADM_s112
                                 + 0.930646 w_k1_AXP_s112
                                 + 0.809802 w_k1_DIS_s112
                                 + 0.849365 w_k1_FDX_s112
                                 + 0.829265 w_k1_IBM_s112
                                 + 0.867403 w_k1_INTC_s112
                                 + 0.890275 w_k1_KO_s112
                                 + 0.945814 w_k1_MAT_s112
                                 + 0.818018 w_k1_MSFT_s112
                                 + 0.769043 w_k1_PG_s112 >= 1
 z_k1_s1123_constraint:          var_1 + z_k1_s1123 + 0.945627 w_k1_ADM_s112
                                 + 0.945062 w_k1_AXP_s112
                                 + 0.627653 w_k1_DIS_s112
                                 + 0.777837 w_k1_FDX_s112 + 1.008 w_k1_IBM_s112
                                 + 0.774345 w_k1_INTC_s112
                                 + 0.894071 w_k1_KO_s112
                                 + 1.11087 w_k1_MAT_s112
                                 + 0.781665 w_k1_MSFT_s112
                                 + 0.828541 w_k1_PG_s112 >= 1
 z_k1_s1124_constraint:          var_1 + z_k1_s1124 + 1.13207 w_k1_ADM_s112
                                 + 1.22272 w_k1_AXP_s112 + 1.0757 w_k1_DIS_s112
                                 + 1.03787 w_k1_FDX_s112
                                 + 1.23293 w_k1_IBM_s112
                                 + 0.902428 w_k1_INTC_s112
                                 + 1.20272 w_k1_KO_s112 + 1.61177 w_k1_MAT_s112
                                 + 0.988786 w_k1_MSFT_s112
                                 + 0.987255 w_k1_PG_s112 >= 1
 z_k1_s1125_constraint:          var_1 + z_k1_s1125 + 0.757319 w_k1_ADM_s112
                                 + 0.649753 w_k1_AXP_s112
                                 + 0.544293 w_k1_DIS_s112
                                 + 0.724264 w_k1_FDX_s112
                                 + 0.667647 w_k1_IBM_s112
                                 + 0.707024 w_k1_INTC_s112
                                 + 0.738711 w_k1_KO_s112
                                 + 0.896116 w_k1_MAT_s112
                                 + 0.732241 w_k1_MSFT_s112
                                 + 0.710318 w_k1_PG_s112 >= 1
 z_k1_s1126_constraint:          var_1 + z_k1_s1126 + 1.0081 w_k1_ADM_s112
                                 + 0.881895 w_k1_AXP_s112
                                 + 0.940581 w_k1_DIS_s112
                                 + 0.933649 w_k1_FDX_s112
                                 + 1.09499 w_k1_IBM_s112
                                 + 0.875799 w_k1_INTC_s112
                                 + 0.969031 w_k1_KO_s112
                                 + 1.15176 w_k1_MAT_s112
                                 + 0.911284 w_k1_MSFT_s112
                                 + 0.874102 w_k1_PG_s112 >= 1
 z_k1_s1127_constraint:          var_1 + z_k1_s1127 + 1.06674 w_k1_ADM_s112
                                 + 1.50948 w_k1_AXP_s112
                                 + 1.79158 w_k1_DIS_s112
                                 + 1.46225 w_k1_FDX_s112 + 1.2142 w_k1_IBM_s112
                                 + 1.59992 w_k1_INTC_s112
                                 + 1.45234 w_k1_KO_s112 + 1.81611 w_k1_MAT_s112
                                 + 1.31202 w_k1_MSFT_s112
                                 + 1.21282 w_k1_PG_s112 >= 1
 z_k1_s1130_constraint:          var_1 + z_k1_s1130 + 0.717796 w_k1_ADM_s113
                                 + 0.766008 w_k1_AXP_s113
                                 + 0.646943 w_k1_DIS_s113
                                 + 0.779828 w_k1_FDX_s113
                                 + 0.950873 w_k1_IBM_s113
                                 + 0.795694 w_k1_INTC_s113
                                 + 0.808609 w_k1_KO_s113 + 1.0211 w_k1_MAT_s113
                                 + 0.74923 w_k1_MSFT_s113
                                 + 0.846377 w_k1_PG_s113 >= 1
 z_k1_s1131_constraint:          var_1 + z_k1_s1131 + 1.0285 w_k1_ADM_s113
                                 + 1.09055 w_k1_AXP_s113
                                 + 0.92808 w_k1_DIS_s113
                                 + 1.01055 w_k1_FDX_s113
                                 + 1.33873 w_k1_IBM_s113
                                 + 1.25508 w_k1_INTC_s113
                                 + 0.986182 w_k1_KO_s113
                                 + 1.10267 w_k1_MAT_s113
                                 + 0.989346 w_k1_MSFT_s113
                                 + 1.00506 w_k1_PG_s113 >= 1
 z_k1_s1132_constraint:          var_1 + z_k1_s1132 + 0.671315 w_k1_ADM_s113
                                 + 0.881737 w_k1_AXP_s113
                                 + 0.714603 w_k1_DIS_s113
                                 + 0.940395 w_k1_FDX_s113
                                 + 0.618085 w_k1_IBM_s113
                                 + 1.06078 w_k1_INTC_s113
                                 + 0.679004 w_k1_KO_s113
                                 + 0.66301 w_k1_MAT_s113
                                 + 0.898966 w_k1_MSFT_s113
                                 + 0.862139 w_k1_PG_s113 >= 1
 z_k1_s1133_constraint:          var_1 + z_k1_s1133 + 0.995172 w_k1_ADM_s113
                                 + 1.2834 w_k1_AXP_s113 + 1.4713 w_k1_DIS_s113
                                 + 1.12842 w_k1_FDX_s113 + 1.7304 w_k1_IBM_s113
                                 + 1.25797 w_k1_INTC_s113
                                 + 1.39036 w_k1_KO_s113 + 1.69296 w_k1_MAT_s113
                                 + 1.18828 w_k1_MSFT_s113
                                 + 1.20072 w_k1_PG_s113 >= 1
 z_k1_s1134_constraint:          var_1 + z_k1_s1134 + 1.00675 w_k1_ADM_s113
                                 + 0.968392 w_k1_AXP_s113
                                 + 0.794129 w_k1_DIS_s113
                                 + 0.988426 w_k1_FDX_s113
                                 + 0.91268 w_k1_IBM_s113
                                 + 1.02886 w_k1_INTC_s113
                                 + 0.840597 w_k1_KO_s113
                                 + 1.13859 w_k1_MAT_s113
                                 + 0.967645 w_k1_MSFT_s113
                                 + 0.994479 w_k1_PG_s113 >= 1
 z_k1_s1135_constraint:          var_1 + z_k1_s1135 + 0.803057 w_k1_ADM_s113
                                 + 0.680735 w_k1_AXP_s113
                                 + 0.747615 w_k1_DIS_s113
                                 + 0.91995 w_k1_FDX_s113
                                 + 0.726553 w_k1_IBM_s113
                                 + 1.14142 w_k1_INTC_s113
                                 + 0.846595 w_k1_KO_s113
                                 + 0.789004 w_k1_MAT_s113
                                 + 0.979379 w_k1_MSFT_s113
                                 + 0.944313 w_k1_PG_s113 >= 1
 z_k1_s1136_constraint:          var_1 + z_k1_s1136 + 1.07711 w_k1_ADM_s113
                                 + 0.971748 w_k1_AXP_s113
                                 + 1.17853 w_k1_DIS_s113
                                 + 1.23633 w_k1_FDX_s113
                                 + 0.611621 w_k1_IBM_s113
                                 + 1.23367 w_k1_INTC_s113
                                 + 0.891445 w_k1_KO_s113
                                 + 0.635698 w_k1_MAT_s113
                                 + 1.10817 w_k1_MSFT_s113
                                 + 1.06877 w_k1_PG_s113 >= 1
 z_k1_s1137_constraint:          var_1 + z_k1_s1137 + 0.988028 w_k1_ADM_s113
                                 + 1.15549 w_k1_AXP_s113
                                 + 1.25223 w_k1_DIS_s113
                                 + 1.23776 w_k1_FDX_s113
                                 + 1.04354 w_k1_IBM_s113
                                 + 1.22538 w_k1_INTC_s113
                                 + 1.01072 w_k1_KO_s113 + 1.09014 w_k1_MAT_s113
                                 + 1.17579 w_k1_MSFT_s113
                                 + 1.19198 w_k1_PG_s113 >= 1
 z_k1_s1140_constraint:          var_1 + z_k1_s1140 + 1.17328 w_k1_ADM_s114
                                 + 1.4629 w_k1_AXP_s114 + 1.44744 w_k1_DIS_s114
                                 + 1.12776 w_k1_FDX_s114
                                 + 1.64149 w_k1_IBM_s114
                                 + 1.29614 w_k1_INTC_s114
                                 + 1.32987 w_k1_KO_s114 + 1.86587 w_k1_MAT_s114
                                 + 1.08223 w_k1_MSFT_s114
                                 + 1.07215 w_k1_PG_s114 >= 1
 z_k1_s1141_constraint:          var_1 + z_k1_s1141 + 1.07717 w_k1_ADM_s114
                                 + 0.908085 w_k1_AXP_s114
                                 + 1.04212 w_k1_DIS_s114
                                 + 1.02563 w_k1_FDX_s114
                                 + 0.698451 w_k1_IBM_s114
                                 + 1.07117 w_k1_INTC_s114
                                 + 0.929071 w_k1_KO_s114
                                 + 0.690968 w_k1_MAT_s114
                                 + 1.00242 w_k1_MSFT_s114
                                 + 1.03031 w_k1_PG_s114 >= 1
 z_k1_s1142_constraint:          var_1 + z_k1_s1142 + 0.744401 w_k1_ADM_s114
                                 + 0.694719 w_k1_AXP_s114
                                 + 0.672618 w_k1_DIS_s114
                                 + 0.764788 w_k1_FDX_s114
                                 + 0.599458 w_k1_IBM_s114
                                 + 0.84761 w_k1_INTC_s114
                                 + 0.703133 w_k1_KO_s114
                                 + 0.495117 w_k1_MAT_s114
                                 + 0.793474 w_k1_MSFT_s114
                                 + 0.747474 w_k1_PG_s114 >= 1
 z_k1_s1143_constraint:          var_1 + z_k1_s1143 + 0.967836 w_k1_ADM_s114
                                 + 0.756098 w_k1_AXP_s114
                                 + 0.649805 w_k1_DIS_s114
                                 + 0.783938 w_k1_FDX_s114
                                 + 0.906499 w_k1_IBM_s114
                                 + 0.922791 w_k1_INTC_s114
                                 + 0.84071 w_k1_KO_s114 + 1.0888 w_k1_MAT_s114
                                 + 0.823535 w_k1_MSFT_s114
                                 + 0.891147 w_k1_PG_s114 >= 1
 z_k1_s1144_constraint:          var_1 + z_k1_s1144 + 1.07771 w_k1_ADM_s114
                                 + 1.09553 w_k1_AXP_s114
                                 + 1.02483 w_k1_DIS_s114
                                 + 0.975498 w_k1_FDX_s114
                                 + 1.27245 w_k1_IBM_s114
                                 + 1.03593 w_k1_INTC_s114
                                 + 1.12159 w_k1_KO_s114 + 1.21985 w_k1_MAT_s114
                                 + 1.00109 w_k1_MSFT_s114
                                 + 1.00372 w_k1_PG_s114 >= 1
 z_k1_s1145_constraint:          var_1 + z_k1_s1145 + 1.25601 w_k1_ADM_s114
                                 + 1.19803 w_k1_AXP_s114
                                 + 1.54991 w_k1_DIS_s114
                                 + 1.27846 w_k1_FDX_s114
                                 + 0.944109 w_k1_IBM_s114
                                 + 1.4165 w_k1_INTC_s114 + 1.23526 w_k1_KO_s114
                                 + 1.30695 w_k1_MAT_s114
                                 + 1.34286 w_k1_MSFT_s114
                                 + 1.12685 w_k1_PG_s114 >= 1
 z_k1_s1146_constraint:          var_1 + z_k1_s1146 + 1.18015 w_k1_ADM_s114
                                 + 1.34624 w_k1_AXP_s114 + 1.0884 w_k1_DIS_s114
                                 + 1.19006 w_k1_FDX_s114
                                 + 0.962117 w_k1_IBM_s114
                                 + 1.16884 w_k1_INTC_s114
                                 + 1.12276 w_k1_KO_s114 + 1.06152 w_k1_MAT_s114
                                 + 1.03355 w_k1_MSFT_s114
                                 + 1.03096 w_k1_PG_s114 >= 1
 z_k1_s1147_constraint:          var_1 + z_k1_s1147 + 0.976969 w_k1_ADM_s114
                                 + 1.1185 w_k1_AXP_s114
                                 + 0.974436 w_k1_DIS_s114
                                 + 1.00914 w_k1_FDX_s114
                                 + 0.848856 w_k1_IBM_s114
                                 + 0.882828 w_k1_INTC_s114
                                 + 0.864415 w_k1_KO_s114
                                 + 1.00405 w_k1_MAT_s114
                                 + 0.896408 w_k1_MSFT_s114
                                 + 0.989387 w_k1_PG_s114 >= 1
 z_k1_s1150_constraint:          var_1 + z_k1_s1150 + 1.10558 w_k1_ADM_s115
                                 + 1.16547 w_k1_AXP_s115
                                 + 1.05083 w_k1_DIS_s115
                                 + 1.01976 w_k1_FDX_s115 + 1.1507 w_k1_IBM_s115
                                 + 1.03098 w_k1_INTC_s115
                                 + 1.14565 w_k1_KO_s115
                                 + 0.966079 w_k1_MAT_s115
                                 + 1.02149 w_k1_MSFT_s115
                                 + 1.23829 w_k1_PG_s115 >= 1
 z_k1_s1151_constraint:          var_1 + z_k1_s1151 + 0.973902 w_k1_ADM_s115
                                 + 0.954214 w_k1_AXP_s115
                                 + 0.925387 w_k1_DIS_s115
                                 + 1.03886 w_k1_FDX_s115
                                 + 0.950936 w_k1_IBM_s115
                                 + 1.16154 w_k1_INTC_s115
                                 + 1.04304 w_k1_KO_s115
                                 + 0.761776 w_k1_MAT_s115
                                 + 0.978149 w_k1_MSFT_s115
                                 + 1.13437 w_k1_PG_s115 >= 1
 z_k1_s1152_constraint:          var_1 + z_k1_s1152 + 0.982617 w_k1_ADM_s115
                                 + 1.4127 w_k1_AXP_s115 + 1.26186 w_k1_DIS_s115
                                 + 1.31017 w_k1_FDX_s115
                                 + 1.13046 w_k1_IBM_s115
                                 + 1.38131 w_k1_INTC_s115
                                 + 1.10014 w_k1_KO_s115 + 1.04711 w_k1_MAT_s115
                                 + 1.25625 w_k1_MSFT_s115
                                 + 1.04448 w_k1_PG_s115 >= 1
 z_k1_s1153_constraint:          var_1 + z_k1_s1153 + 1.15528 w_k1_ADM_s115
                                 + 1.41284 w_k1_AXP_s115
                                 + 1.07859 w_k1_DIS_s115
                                 + 0.963313 w_k1_FDX_s115
                                 + 1.63554 w_k1_IBM_s115
                                 + 1.13945 w_k1_INTC_s115
                                 + 1.28344 w_k1_KO_s115 + 1.38303 w_k1_MAT_s115
                                 + 0.98551 w_k1_MSFT_s115
                                 + 1.19718 w_k1_PG_s115 >= 1
 z_k1_s1154_constraint:          var_1 + z_k1_s1154 + 0.809549 w_k1_ADM_s115
                                 + 0.776413 w_k1_AXP_s115
                                 + 0.781001 w_k1_DIS_s115
                                 + 0.865835 w_k1_FDX_s115
                                 + 0.694915 w_k1_IBM_s115
                                 + 1.02813 w_k1_INTC_s115
                                 + 0.73319 w_k1_KO_s115
                                 + 0.589052 w_k1_MAT_s115
                                 + 0.964503 w_k1_MSFT_s115
                                 + 0.815867 w_k1_PG_s115 >= 1
 z_k1_s1155_constraint:          var_1 + z_k1_s1155 + 1.12087 w_k1_ADM_s115
                                 + 0.991133 w_k1_AXP_s115
                                 + 1.03467 w_k1_DIS_s115
                                 + 1.18196 w_k1_FDX_s115
                                 + 0.82941 w_k1_IBM_s115
                                 + 1.34445 w_k1_INTC_s115
                                 + 1.01947 w_k1_KO_s115
                                 + 0.635194 w_k1_MAT_s115
                                 + 1.26615 w_k1_MSFT_s115
                                 + 1.14968 w_k1_PG_s115 >= 1
 z_k1_s1156_constraint:          var_1 + z_k1_s1156 + 1.01732 w_k1_ADM_s115
                                 + 1.41137 w_k1_AXP_s115
                                 + 1.83476 w_k1_DIS_s115
                                 + 1.28345 w_k1_FDX_s115
                                 + 1.44503 w_k1_IBM_s115
                                 + 1.40665 w_k1_INTC_s115
                                 + 1.53664 w_k1_KO_s115 + 1.45077 w_k1_MAT_s115
                                 + 1.44575 w_k1_MSFT_s115
                                 + 1.24518 w_k1_PG_s115 >= 1
 z_k1_s1157_constraint:          var_1 + z_k1_s1157 + 0.842773 w_k1_ADM_s115
                                 + 0.800044 w_k1_AXP_s115
                                 + 0.704096 w_k1_DIS_s115
                                 + 0.79068 w_k1_FDX_s115 + 1.0365 w_k1_IBM_s115
                                 + 0.871199 w_k1_INTC_s115
                                 + 0.885596 w_k1_KO_s115
                                 + 0.802619 w_k1_MAT_s115
                                 + 0.826137 w_k1_MSFT_s115
                                 + 0.949463 w_k1_PG_s115 >= 1
 z_k1_s1160_constraint:          var_1 + z_k1_s1160 + 1.36393 w_k1_ADM_s116
                                 + 1.5939 w_k1_AXP_s116 + 1.41563 w_k1_DIS_s116
                                 + 1.34712 w_k1_FDX_s116
                                 + 1.05048 w_k1_IBM_s116
                                 + 0.993337 w_k1_INTC_s116
                                 + 1.0451 w_k1_KO_s116 + 0.868307 w_k1_MAT_s116
                                 + 1.18204 w_k1_MSFT_s116
                                 + 1.00224 w_k1_PG_s116 >= 1
 z_k1_s1161_constraint:          var_1 + z_k1_s1161 + 1.05395 w_k1_ADM_s116
                                 + 0.891437 w_k1_AXP_s116
                                 + 0.820992 w_k1_DIS_s116
                                 + 0.856054 w_k1_FDX_s116
                                 + 0.633356 w_k1_IBM_s116
                                 + 0.81068 w_k1_INTC_s116
                                 + 0.859292 w_k1_KO_s116
                                 + 0.679618 w_k1_MAT_s116
                                 + 0.872541 w_k1_MSFT_s116
                                 + 0.916268 w_k1_PG_s116 >= 1
 z_k1_s1162_constraint:          var_1 + z_k1_s1162 + 0.989581 w_k1_ADM_s116
                                 + 0.681386 w_k1_AXP_s116
                                 + 0.953854 w_k1_DIS_s116
                                 + 0.791241 w_k1_FDX_s116
                                 + 1.08671 w_k1_IBM_s116
                                 + 0.798807 w_k1_INTC_s116
                                 + 1.09548 w_k1_KO_s116
                                 + 0.741632 w_k1_MAT_s116
                                 + 0.87388 w_k1_MSFT_s116
                                 + 0.907932 w_k1_PG_s116 >= 1
 z_k1_s1163_constraint:          var_1 + z_k1_s1163 + 0.975931 w_k1_ADM_s116
                                 + 1.02613 w_k1_AXP_s116
                                 + 0.910374 w_k1_DIS_s116
                                 + 0.94788 w_k1_FDX_s116
                                 + 1.08642 w_k1_IBM_s116
                                 + 0.956028 w_k1_INTC_s116
                                 + 1.01309 w_k1_KO_s116
                                 + 0.767294 w_k1_MAT_s116
                                 + 0.896918 w_k1_MSFT_s116
                                 + 0.925714 w_k1_PG_s116 >= 1
 z_k1_s1164_constraint:          var_1 + z_k1_s1164 + 1.04272 w_k1_ADM_s116
                                 + 1.12103 w_k1_AXP_s116
                                 + 1.26307 w_k1_DIS_s116
                                 + 1.03881 w_k1_FDX_s116
                                 + 1.50058 w_k1_IBM_s116
                                 + 0.913953 w_k1_INTC_s116
                                 + 1.17044 w_k1_KO_s116 + 1.0993 w_k1_MAT_s116
                                 + 1.06384 w_k1_MSFT_s116
                                 + 1.00787 w_k1_PG_s116 >= 1
 z_k1_s1165_constraint:          var_1 + z_k1_s1165 + 1.5258 w_k1_ADM_s116
                                 + 1.30197 w_k1_AXP_s116
                                 + 1.37285 w_k1_DIS_s116
                                 + 1.19097 w_k1_FDX_s116
                                 + 1.22268 w_k1_IBM_s116
                                 + 1.03119 w_k1_INTC_s116
                                 + 1.23737 w_k1_KO_s116 + 0.9115 w_k1_MAT_s116
                                 + 1.15103 w_k1_MSFT_s116
                                 + 1.13274 w_k1_PG_s116 >= 1
 z_k1_s1166_constraint:          var_1 + z_k1_s1166 + 1.04842 w_k1_ADM_s116
                                 + 0.911959 w_k1_AXP_s116
                                 + 0.76094 w_k1_DIS_s116
                                 + 0.894764 w_k1_FDX_s116
                                 + 0.822033 w_k1_IBM_s116
                                 + 0.752442 w_k1_INTC_s116
                                 + 0.809743 w_k1_KO_s116
                                 + 0.583307 w_k1_MAT_s116
                                 + 0.852579 w_k1_MSFT_s116
                                 + 0.91063 w_k1_PG_s116 >= 1
 z_k1_s1167_constraint:          var_1 + z_k1_s1167 + 1.17246 w_k1_ADM_s116
                                 + 1.05153 w_k1_AXP_s116
                                 + 1.29714 w_k1_DIS_s116
                                 + 1.22659 w_k1_FDX_s116
                                 + 0.861478 w_k1_IBM_s116
                                 + 1.02405 w_k1_INTC_s116
                                 + 1.00943 w_k1_KO_s116
                                 + 0.550367 w_k1_MAT_s116
                                 + 1.17596 w_k1_MSFT_s116
                                 + 1.14253 w_k1_PG_s116 >= 1
 z_k1_s1170_constraint:          var_1 + z_k1_s1170 + 0.973126 w_k1_ADM_s117
                                 + 0.875003 w_k1_AXP_s117
                                 + 0.786076 w_k1_DIS_s117
                                 + 0.914291 w_k1_FDX_s117
                                 + 0.55149 w_k1_IBM_s117
                                 + 0.682298 w_k1_INTC_s117
                                 + 0.725304 w_k1_KO_s117
                                 + 0.590724 w_k1_MAT_s117
                                 + 0.816934 w_k1_MSFT_s117
                                 + 0.903053 w_k1_PG_s117 >= 1
 z_k1_s1171_constraint:          var_1 + z_k1_s1171 + 1.22828 w_k1_ADM_s117
                                 + 1.10086 w_k1_AXP_s117
                                 + 1.02562 w_k1_DIS_s117
                                 + 0.970951 w_k1_FDX_s117
                                 + 1.0462 w_k1_IBM_s117
                                 + 0.864674 w_k1_INTC_s117
                                 + 1.09987 w_k1_KO_s117 + 1.1652 w_k1_MAT_s117
                                 + 0.917607 w_k1_MSFT_s117
                                 + 1.05996 w_k1_PG_s117 >= 1
 z_k1_s1172_constraint:          var_1 + z_k1_s1172 + 1.14857 w_k1_ADM_s117
                                 + 0.963567 w_k1_AXP_s117
                                 + 1.15422 w_k1_DIS_s117
                                 + 1.11109 w_k1_FDX_s117
                                 + 0.81055 w_k1_IBM_s117
                                 + 1.04575 w_k1_INTC_s117
                                 + 1.10112 w_k1_KO_s117
                                 + 0.897975 w_k1_MAT_s117
                                 + 1.15368 w_k1_MSFT_s117
                                 + 1.21013 w_k1_PG_s117 >= 1
 z_k1_s1173_constraint:          var_1 + z_k1_s1173 + 1.21963 w_k1_ADM_s117
                                 + 1.51058 w_k1_AXP_s117
                                 + 1.17758 w_k1_DIS_s117
                                 + 1.31155 w_k1_FDX_s117
                                 + 0.707666 w_k1_IBM_s117
                                 + 0.900889 w_k1_INTC_s117
                                 + 1.09241 w_k1_KO_s117
                                 + 0.840798 w_k1_MAT_s117
                                 + 1.106 w_k1_MSFT_s117 + 1.18884 w_k1_PG_s117
                                 >= 1
 z_k1_s1174_constraint:          var_1 + z_k1_s1174 + 1.25483 w_k1_ADM_s117
                                 + 1.75008 w_k1_AXP_s117
                                 + 1.37863 w_k1_DIS_s117
                                 + 1.16797 w_k1_FDX_s117
                                 + 1.45086 w_k1_IBM_s117
                                 + 0.932624 w_k1_INTC_s117
                                 + 1.35537 w_k1_KO_s117 + 1.35978 w_k1_MAT_s117
                                 + 1.11537 w_k1_MSFT_s117
                                 + 1.19245 w_k1_PG_s117 >= 1
 z_k1_s1175_constraint:          var_1 + z_k1_s1175 + 0.819299 w_k1_ADM_s117
                                 + 0.856674 w_k1_AXP_s117
                                 + 0.663749 w_k1_DIS_s117
                                 + 0.731846 w_k1_FDX_s117
                                 + 0.857514 w_k1_IBM_s117
                                 + 0.728557 w_k1_INTC_s117
                                 + 0.869361 w_k1_KO_s117
                                 + 0.767074 w_k1_MAT_s117
                                 + 0.746275 w_k1_MSFT_s117
                                 + 0.991763 w_k1_PG_s117 >= 1
 z_k1_s1176_constraint:          var_1 + z_k1_s1176 + 1.12949 w_k1_ADM_s117
                                 + 0.846552 w_k1_AXP_s117
                                 + 1.33124 w_k1_DIS_s117
                                 + 0.891771 w_k1_FDX_s117
                                 + 1.51143 w_k1_IBM_s117
                                 + 0.850954 w_k1_INTC_s117
                                 + 1.36645 w_k1_KO_s117 + 1.19805 w_k1_MAT_s117
                                 + 1.06133 w_k1_MSFT_s117 + 1.0585 w_k1_PG_s117
                                 >= 1
 z_k1_s1177_constraint:          var_1 + z_k1_s1177 + 1.16667 w_k1_ADM_s117
                                 + 1.01326 w_k1_AXP_s117
                                 + 0.702733 w_k1_DIS_s117
                                 + 0.866098 w_k1_FDX_s117
                                 + 0.74694 w_k1_IBM_s117
                                 + 0.829023 w_k1_INTC_s117
                                 + 0.824083 w_k1_KO_s117
                                 + 0.70851 w_k1_MAT_s117
                                 + 0.749434 w_k1_MSFT_s117
                                 + 0.948265 w_k1_PG_s117 >= 1
 k1_return_constraint:           0.0530308382159439 w_k1_ADM_s000
                                 + 0.0137166178733611 w_k1_ADM_s001
                                 + 0.0173321451684196 w_k1_ADM_s002
                                 + 0.0225249907819394 w_k1_ADM_s003
                                 + 0.0184742616489238 w_k1_ADM_s004
                                 + 0.0117479423325761 w_k1_ADM_s005
                                 + 0.0110177853145946 w_k1_ADM_s006
                                 + 0.0507257357659138 w_k1_ADM_s007
                                 + 0.0384075690107744 w_k1_ADM_s010
                                 + 0.027467630312659 w_k1_ADM_s011
                                 + 0.0507679928497359 w_k1_ADM_s012
                                 + 0.0247771931371724 w_k1_ADM_s013
                                 + 0.14010161313121 w_k1_ADM_s014
                                 + 0.0360849287576299 w_k1_ADM_s015
                                 + 0.0263011198529809 w_k1_ADM_s016
                                 + 0.0479866834728268 w_k1_ADM_s017
                                 + 0.0227366320934176 w_k1_ADM_s100
                                 + 0.0092166668593454 w_k1_ADM_s101
                                 + 0.0222862719704981 w_k1_ADM_s102
                                 + 0.0299630249921764 w_k1_ADM_s103
                                 + 0.0307202751014979 w_k1_ADM_s104
                                 + 0.0123082550642461 w_k1_ADM_s105
                                 + 0.0135451663333106 w_k1_ADM_s106
                                 + 0.0182501261286091 w_k1_ADM_s107
                                 + 0.0207302272583034 w_k1_ADM_s110
                                 + 0.0312292562261985 w_k1_ADM_s111
                                 + 0.0207386831990523 w_k1_ADM_s112
                                 + 0.0708217831479932 w_k1_ADM_s113
                                 + 0.0219787777984848 w_k1_ADM_s114
                                 + 0.0218489718879235 w_k1_ADM_s115
                                 + 0.0387673305935602 w_k1_ADM_s116
                                 + 0.0563139983528287 w_k1_ADM_s117
                                 + 0.0556897816859931 w_k1_AXP_s000
                                 + 0.0139854113006561 w_k1_AXP_s001
                                 + 0.0175373080533955 w_k1_AXP_s002
                                 + 0.0223247491390332 w_k1_AXP_s003
                                 + 0.0161018949147857 w_k1_AXP_s004
                                 + 0.0128998428746318 w_k1_AXP_s005
                                 + 0.0126675111840718 w_k1_AXP_s006
                                 + 0.0526914828085251 w_k1_AXP_s007
                                 + 0.0368113945508462 w_k1_AXP_s010
                                 + 0.029176210519579 w_k1_AXP_s011
                                 + 0.0519657954020281 w_k1_AXP_s012
                                 + 0.0280361306288848 w_k1_AXP_s013
                                 + 0.126559608197669 w_k1_AXP_s014
                                 + 0.0354009600850195 w_k1_AXP_s015
                                 + 0.0263848056408074 w_k1_AXP_s016
                                 + 0.0456124931981147 w_k1_AXP_s017
                                 + 0.0225120776329479 w_k1_AXP_s100
                                 + 0.00772827980560801 w_k1_AXP_s101
                                 + 0.0226443374228893 w_k1_AXP_s102
                                 + 0.0302751793531903 w_k1_AXP_s103
                                 + 0.0311047573193513 w_k1_AXP_s104
                                 + 0.0132031141155106 w_k1_AXP_s105
                                 + 0.014644858270995 w_k1_AXP_s106
                                 + 0.019242026578782 w_k1_AXP_s107
                                 + 0.0188410618597024 w_k1_AXP_s110
                                 + 0.0374403194913305 w_k1_AXP_s111
                                 + 0.0236992447601586 w_k1_AXP_s112
                                 + 0.0767533122213986 w_k1_AXP_s113
                                 + 0.0217001187989866 w_k1_AXP_s114
                                 + 0.0232325546913835 w_k1_AXP_s115
                                 + 0.0368063087759619 w_k1_AXP_s116
                                 + 0.0583244765996149 w_k1_AXP_s117
                                 + 0.0564344608315055 w_k1_DIS_s000
                                 + 0.013921029515679 w_k1_DIS_s001
                                 + 0.0166896066837229 w_k1_DIS_s002
                                 + 0.0232247043886483 w_k1_DIS_s003
                                 + 0.0160774447785113 w_k1_DIS_s004
                                 + 0.0126928888573156 w_k1_DIS_s005
                                 + 0.0115018884190621 w_k1_DIS_s006
                                 + 0.046347837521206 w_k1_DIS_s007
                                 + 0.0371059376278789 w_k1_DIS_s010
                                 + 0.0293403814245898 w_k1_DIS_s011
                                 + 0.0530937298647432 w_k1_DIS_s012
                                 + 0.0256065185533012 w_k1_DIS_s013
                                 + 0.126838846178101 w_k1_DIS_s014
                                 + 0.0387067111519588 w_k1_DIS_s015
                                 + 0.0261055392850079 w_k1_DIS_s016
                                 + 0.0489597384553066 w_k1_DIS_s017
                                 + 0.0240478585078344 w_k1_DIS_s100
                                 + 0.00882751791981228 w_k1_DIS_s101
                                 + 0.0196194980836001 w_k1_DIS_s102
                                 + 0.0279016801025089 w_k1_DIS_s103
                                 + 0.0321809506091179 w_k1_DIS_s104
                                 + 0.0123936719061553 w_k1_DIS_s105
                                 + 0.0145544801915862 w_k1_DIS_s106
                                 + 0.0166815240668421 w_k1_DIS_s107
                                 + 0.0175629319073607 w_k1_DIS_s110
                                 + 0.034937429889904 w_k1_DIS_s111
                                 + 0.0221194585971112 w_k1_DIS_s112
                                 + 0.0761283038419125 w_k1_DIS_s113
                                 + 0.0211640823618665 w_k1_DIS_s114
                                 + 0.0220276515062459 w_k1_DIS_s115
                                 + 0.0368017408769654 w_k1_DIS_s116
                                 + 0.0515189622451851 w_k1_DIS_s117
                                 + 0.0536955162102369 w_k1_FDX_s000
                                 + 0.0133360033838148 w_k1_FDX_s001
                                 + 0.0163628836441112 w_k1_FDX_s002
                                 + 0.0222358849237763 w_k1_FDX_s003
                                 + 0.0159379426887807 w_k1_FDX_s004
                                 + 0.0119284522839611 w_k1_FDX_s005
                                 + 0.0120072610246251 w_k1_FDX_s006
                                 + 0.0454955433731278 w_k1_FDX_s007
                                 + 0.0375747730135481 w_k1_FDX_s010
                                 + 0.0292008722224416 w_k1_FDX_s011
                                 + 0.052944845221902 w_k1_FDX_s012
                                 + 0.0255370731203227 w_k1_FDX_s013
                                 + 0.132946582133132 w_k1_FDX_s014
                                 + 0.0379218565776888 w_k1_FDX_s015
                                 + 0.0276687364505076 w_k1_FDX_s016
                                 + 0.0484535522897615 w_k1_FDX_s017
                                 + 0.022324928739822 w_k1_FDX_s100
                                 + 0.00836894396820539 w_k1_FDX_s101
                                 + 0.0212208521128505 w_k1_FDX_s102
                                 + 0.0282972083759501 w_k1_FDX_s103
                                 + 0.0301380568707446 w_k1_FDX_s104
                                 + 0.0128212702102884 w_k1_FDX_s105
                                 + 0.0142004554005041 w_k1_FDX_s106
                                 + 0.0175822372803218 w_k1_FDX_s107
                                 + 0.0191580471079851 w_k1_FDX_s110
                                 + 0.0320717967859117 w_k1_FDX_s111
                                 + 0.0222015785837446 w_k1_FDX_s112
                                 + 0.0797588404177566 w_k1_FDX_s113
                                 + 0.0207959251155744 w_k1_FDX_s114
                                 + 0.0223731916407979 w_k1_FDX_s115
                                 + 0.0353016666710188 w_k1_FDX_s116
                                 + 0.0497062330054625 w_k1_FDX_s117
                                 + 0.0536153249709208 w_k1_IBM_s000
                                 + 0.0129095179870707 w_k1_IBM_s001
                                 + 0.0167503537785918 w_k1_IBM_s002
                                 + 0.0223787385326852 w_k1_IBM_s003
                                 + 0.0154061275152131 w_k1_IBM_s004
                                 + 0.0107581957220776 w_k1_IBM_s005
                                 + 0.0121918030061033 w_k1_IBM_s006
                                 + 0.0472244671678755 w_k1_IBM_s007
                                 + 0.0378404585077331 w_k1_IBM_s010
                                 + 0.0289583493006656 w_k1_IBM_s011
                                 + 0.053487241089138 w_k1_IBM_s012
                                 + 0.027121823528406 w_k1_IBM_s013
                                 + 0.131416979819304 w_k1_IBM_s014
                                 + 0.0384315913954792 w_k1_IBM_s015
                                 + 0.0282107023994932 w_k1_IBM_s016
                                 + 0.0509679533317047 w_k1_IBM_s017
                                 + 0.0225461119896194 w_k1_IBM_s100
                                 + 0.00877451263149397 w_k1_IBM_s101
                                 + 0.0216655689831247 w_k1_IBM_s102
                                 + 0.0288967019412482 w_k1_IBM_s103
                                 + 0.02814859692051 w_k1_IBM_s104
                                 + 0.013111505961075 w_k1_IBM_s105
                                 + 0.0132989002549186 w_k1_IBM_s106
                                 + 0.0174084804039046 w_k1_IBM_s107
                                 + 0.0193698619985685 w_k1_IBM_s110
                                 + 0.0320236789170436 w_k1_IBM_s111
                                 + 0.0223978905741009 w_k1_IBM_s112
                                 + 0.0797794915236975 w_k1_IBM_s113
                                 + 0.0206577437952988 w_k1_IBM_s114
                                 + 0.0241570809004921 w_k1_IBM_s115
                                 + 0.0347889865258609 w_k1_IBM_s116
                                 + 0.0502951555356346 w_k1_IBM_s117
                                 + 0.047831589996138 w_k1_INTC_s000
                                 + 0.0121909920345595 w_k1_INTC_s001
                                 + 0.0192902102999259 w_k1_INTC_s002
                                 + 0.0210077692200195 w_k1_INTC_s003
                                 + 0.0151065277918742 w_k1_INTC_s004
                                 + 0.0110759942239346 w_k1_INTC_s005
                                 + 0.011721057982697 w_k1_INTC_s006
                                 + 0.0451539115621266 w_k1_INTC_s007
                                 + 0.0351858732364617 w_k1_INTC_s010
                                 + 0.0261422666654296 w_k1_INTC_s011
                                 + 0.0515470509760949 w_k1_INTC_s012
                                 + 0.0241318223789587 w_k1_INTC_s013
                                 + 0.110116630744104 w_k1_INTC_s014
                                 + 0.0404910963400646 w_k1_INTC_s015
                                 + 0.0290628043793574 w_k1_INTC_s016
                                 + 0.0459738641940886 w_k1_INTC_s017
                                 + 0.0226316894497634 w_k1_INTC_s100
                                 + 0.00781167154767792 w_k1_INTC_s101
                                 + 0.0229326501439757 w_k1_INTC_s102
                                 + 0.0262822491523326 w_k1_INTC_s103
                                 + 0.0279713117008665 w_k1_INTC_s104
                                 + 0.0124372684409851 w_k1_INTC_s105
                                 + 0.0121254581726981 w_k1_INTC_s106
                                 + 0.0190788333946368 w_k1_INTC_s107
                                 + 0.018129721528573 w_k1_INTC_s110
                                 + 0.0319533068174949 w_k1_INTC_s111
                                 + 0.0220951819372582 w_k1_INTC_s112
                                 + 0.086421908013075 w_k1_INTC_s113
                                 + 0.0220914880058756 w_k1_INTC_s114
                                 + 0.0248492786912527 w_k1_INTC_s115
                                 + 0.0309313379464034 w_k1_INTC_s116
                                 + 0.0429370988428493 w_k1_INTC_s117
                                 + 0.0539370704924925 w_k1_KO_s000
                                 + 0.0130058776881533 w_k1_KO_s001
                                 + 0.0146566524502107 w_k1_KO_s002
                                 + 0.0220715616146276 w_k1_KO_s003
                                 + 0.0154569499481434 w_k1_KO_s004
                                 + 0.0112644926541373 w_k1_KO_s005
                                 + 0.0108926535421788 w_k1_KO_s006
                                 + 0.0459495891873479 w_k1_KO_s007
                                 + 0.0360196975051406 w_k1_KO_s010
                                 + 0.0274907189577504 w_k1_KO_s011
                                 + 0.0538389711339858 w_k1_KO_s012
                                 + 0.027393089824627 w_k1_KO_s013
                                 + 0.118776146103194 w_k1_KO_s014
                                 + 0.0370327153568718 w_k1_KO_s015
                                 + 0.0294124473005637 w_k1_KO_s016
                                 + 0.0519717114506474 w_k1_KO_s017
                                 + 0.0233514230415643 w_k1_KO_s100
                                 + 0.00840663084543943 w_k1_KO_s101
                                 + 0.0199524854716534 w_k1_KO_s102
                                 + 0.025899887658262 w_k1_KO_s103
                                 + 0.0269558978829416 w_k1_KO_s104
                                 + 0.0136783376196233 w_k1_KO_s105
                                 + 0.0122876528757404 w_k1_KO_s106
                                 + 0.0168412604230727 w_k1_KO_s107
                                 + 0.0210356354883669 w_k1_KO_s110
                                 + 0.0324739096862023 w_k1_KO_s111
                                 + 0.0225166625148821 w_k1_KO_s112
                                 + 0.0730965701080885 w_k1_KO_s113
                                 + 0.0210619810675396 w_k1_KO_s114
                                 + 0.0233351687612944 w_k1_KO_s115
                                 + 0.0344338948084981 w_k1_KO_s116
                                 + 0.0534578313531316 w_k1_KO_s117
                                 + 0.0523503992200255 w_k1_MAT_s000
                                 + 0.0128052982725166 w_k1_MAT_s001
                                 + 0.0141339138687654 w_k1_MAT_s002
                                 + 0.0219268205120657 w_k1_MAT_s003
                                 + 0.0180932796195977 w_k1_MAT_s004
                                 + 0.010591286141655 w_k1_MAT_s005
                                 + 0.0117861526773074 w_k1_MAT_s006
                                 + 0.0509561200442198 w_k1_MAT_s007
                                 + 0.0329935093484997 w_k1_MAT_s010
                                 + 0.0362549570066014 w_k1_MAT_s011
                                 + 0.0521995528843233 w_k1_MAT_s012
                                 + 0.0329349689333936 w_k1_MAT_s013
                                 + 0.138018681032123 w_k1_MAT_s014
                                 + 0.0334194743617972 w_k1_MAT_s015
                                 + 0.0278958221581513 w_k1_MAT_s016
                                 + 0.0587342901115316 w_k1_MAT_s017
                                 + 0.0241130474221223 w_k1_MAT_s100
                                 + 0.00712440835946508 w_k1_MAT_s101
                                 + 0.016199181079724 w_k1_MAT_s102
                                 + 0.0256768693181913 w_k1_MAT_s103
                                 + 0.0314095475442408 w_k1_MAT_s104
                                 + 0.01491622929838 w_k1_MAT_s105
                                 + 0.0146979040508869 w_k1_MAT_s106
                                 + 0.0166128471788706 w_k1_MAT_s107
                                 + 0.018927695350195 w_k1_MAT_s110
                                 + 0.0309060266252998 w_k1_MAT_s111
                                 + 0.0282764502200034 w_k1_MAT_s112
                                 + 0.082635785576693 w_k1_MAT_s113
                                 + 0.0228122566819717 w_k1_MAT_s114
                                 + 0.0202102363396998 w_k1_MAT_s115
                                 + 0.0260427960979697 w_k1_MAT_s116
                                 + 0.0499559345266264 w_k1_MAT_s117
                                 + 0.0533768728923597 w_k1_MSFT_s000
                                 + 0.0126510790836529 w_k1_MSFT_s001
                                 + 0.0167694331301773 w_k1_MSFT_s002
                                 + 0.0227492784947756 w_k1_MSFT_s003
                                 + 0.0158640481813846 w_k1_MSFT_s004
                                 + 0.0117686914274163 w_k1_MSFT_s005
                                 + 0.0121740473302486 w_k1_MSFT_s006
                                 + 0.0436091231212448 w_k1_MSFT_s007
                                 + 0.0381244576889254 w_k1_MSFT_s010
                                 + 0.0283368795499635 w_k1_MSFT_s011
                                 + 0.0561133405825195 w_k1_MSFT_s012
                                 + 0.0272697259572079 w_k1_MSFT_s013
                                 + 0.119479130139769 w_k1_MSFT_s014
                                 + 0.0409957910252487 w_k1_MSFT_s015
                                 + 0.0276152864479648 w_k1_MSFT_s016
                                 + 0.0543470447372633 w_k1_MSFT_s017
                                 + 0.0236343130396035 w_k1_MSFT_s100
                                 + 0.00844634863994238 w_k1_MSFT_s101
                                 + 0.0208593786596985 w_k1_MSFT_s102
                                 + 0.0249641992301106 w_k1_MSFT_s103
                                 + 0.0299710656993747 w_k1_MSFT_s104
                                 + 0.0139061012824264 w_k1_MSFT_s105
                                 + 0.0130110503719288 w_k1_MSFT_s106
                                 + 0.0176518548728303 w_k1_MSFT_s107
                                 + 0.0174991963592001 w_k1_MSFT_s110
                                 + 0.0306677372689667 w_k1_MSFT_s111
                                 + 0.020944251821941 w_k1_MSFT_s112
                                 + 0.0784129749051144 w_k1_MSFT_s113
                                 + 0.0206002864214115 w_k1_MSFT_s114
                                 + 0.0230563862459061 w_k1_MSFT_s115
                                 + 0.0340727474636834 w_k1_MSFT_s116
                                 + 0.0477474805242817 w_k1_MSFT_s117
                                 + 0.0556192477160733 w_k1_PG_s000
                                 + 0.0129581459433531 w_k1_PG_s001
                                 + 0.01580419470329 w_k1_PG_s002
                                 + 0.0234953241655697 w_k1_PG_s003
                                 + 0.0149957301884283 w_k1_PG_s004
                                 + 0.0110943845971606 w_k1_PG_s005
                                 + 0.0119853212572263 w_k1_PG_s006
                                 + 0.0459793148045407 w_k1_PG_s007
                                 + 0.0361230344829182 w_k1_PG_s010
                                 + 0.0268804838192167 w_k1_PG_s011
                                 + 0.0547747789789922 w_k1_PG_s012
                                 + 0.0254243579313876 w_k1_PG_s013
                                 + 0.130080451205969 w_k1_PG_s014
                                 + 0.0373497503834539 w_k1_PG_s015
                                 + 0.0287950711313756 w_k1_PG_s016
                                 + 0.0477919908561277 w_k1_PG_s017
                                 + 0.020756529646472 w_k1_PG_s100
                                 + 0.00811824027103825 w_k1_PG_s101
                                 + 0.0203124610682036 w_k1_PG_s102
                                 + 0.0262943385690195 w_k1_PG_s103
                                 + 0.0282874825586052 w_k1_PG_s104
                                 + 0.0133922339596365 w_k1_PG_s105
                                 + 0.0127417229787599 w_k1_PG_s106
                                 + 0.0164278109625471 w_k1_PG_s107
                                 + 0.0181615536430291 w_k1_PG_s110
                                 + 0.0336674608321814 w_k1_PG_s111
                                 + 0.0201097228714814 w_k1_PG_s112
                                 + 0.0793565983450743 w_k1_PG_s113
                                 + 0.0204824143063557 w_k1_PG_s114
                                 + 0.0240496554832531 w_k1_PG_s115
                                 + 0.0336598926855969 w_k1_PG_s116
                                 + 0.0539833909007528 w_k1_PG_s117 >= 1.2
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
