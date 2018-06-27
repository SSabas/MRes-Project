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
 w_k1_ADM_s0_return_constraint:  1.11132 w_0_ADM - w_k1_ADM_s0
                                 + 0.99 b_k1_ADM_s0 - 1.01 s_k1_ADM_s0  = 0
 w_k1_ADM_s1_return_constraint:  0.912237 w_0_ADM - w_k1_ADM_s1
                                 + 0.99 b_k1_ADM_s1 - 1.01 s_k1_ADM_s1  = 0
 w_k1_ADM_s00_return_constraint: 1.08955 w_k1_ADM_s0 - w_k1_ADM_s00
                                 + 0.99 b_k1_ADM_s00 - 1.01 s_k1_ADM_s00  = 0
 w_k1_ADM_s01_return_constraint: 0.927875 w_k1_ADM_s0 - w_k1_ADM_s01
                                 + 0.99 b_k1_ADM_s01 - 1.01 s_k1_ADM_s01  = 0
 w_k1_ADM_s10_return_constraint: 1.27136 w_k1_ADM_s1 - w_k1_ADM_s10
                                 + 0.99 b_k1_ADM_s10 - 1.01 s_k1_ADM_s10  = 0
 w_k1_ADM_s11_return_constraint: 0.899785 w_k1_ADM_s1 - w_k1_ADM_s11
                                 + 0.99 b_k1_ADM_s11 - 1.01 s_k1_ADM_s11  = 0
 w_k1_ADM_s000_return_constraint: 0.978738 w_k1_ADM_s00 - w_k1_ADM_s000
                                 + 0.99 b_k1_ADM_s000 - 1.01 s_k1_ADM_s000  = 0
 w_k1_ADM_s001_return_constraint: 1.07538 w_k1_ADM_s00 - w_k1_ADM_s001
                                 + 0.99 b_k1_ADM_s001 - 1.01 s_k1_ADM_s001  = 0
 w_k1_ADM_s002_return_constraint: 0.644278 w_k1_ADM_s00 - w_k1_ADM_s002
                                 + 0.99 b_k1_ADM_s002 - 1.01 s_k1_ADM_s002  = 0
 w_k1_ADM_s003_return_constraint: 1.10142 w_k1_ADM_s00 - w_k1_ADM_s003
                                 + 0.99 b_k1_ADM_s003 - 1.01 s_k1_ADM_s003  = 0
 w_k1_ADM_s004_return_constraint: 1.53346 w_k1_ADM_s00 - w_k1_ADM_s004
                                 + 0.99 b_k1_ADM_s004 - 1.01 s_k1_ADM_s004  = 0
 w_k1_ADM_s005_return_constraint: 0.911506 w_k1_ADM_s00 - w_k1_ADM_s005
                                 + 0.99 b_k1_ADM_s005 - 1.01 s_k1_ADM_s005  = 0
 w_k1_ADM_s006_return_constraint: 1.09285 w_k1_ADM_s00 - w_k1_ADM_s006
                                 + 0.99 b_k1_ADM_s006 - 1.01 s_k1_ADM_s006  = 0
 w_k1_ADM_s007_return_constraint: 0.886307 w_k1_ADM_s00 - w_k1_ADM_s007
                                 + 0.99 b_k1_ADM_s007 - 1.01 s_k1_ADM_s007  = 0
 w_k1_ADM_s010_return_constraint: 1.2696 w_k1_ADM_s01 - w_k1_ADM_s010
                                 + 0.99 b_k1_ADM_s010 - 1.01 s_k1_ADM_s010  = 0
 w_k1_ADM_s011_return_constraint: 0.854718 w_k1_ADM_s01 - w_k1_ADM_s011
                                 + 0.99 b_k1_ADM_s011 - 1.01 s_k1_ADM_s011  = 0
 w_k1_ADM_s012_return_constraint: 0.94592 w_k1_ADM_s01 - w_k1_ADM_s012
                                 + 0.99 b_k1_ADM_s012 - 1.01 s_k1_ADM_s012  = 0
 w_k1_ADM_s013_return_constraint: 0.826005 w_k1_ADM_s01 - w_k1_ADM_s013
                                 + 0.99 b_k1_ADM_s013 - 1.01 s_k1_ADM_s013  = 0
 w_k1_ADM_s014_return_constraint: 0.903701 w_k1_ADM_s01 - w_k1_ADM_s014
                                 + 0.99 b_k1_ADM_s014 - 1.01 s_k1_ADM_s014  = 0
 w_k1_ADM_s015_return_constraint: 1.03597 w_k1_ADM_s01 - w_k1_ADM_s015
                                 + 0.99 b_k1_ADM_s015 - 1.01 s_k1_ADM_s015  = 0
 w_k1_ADM_s016_return_constraint: 1.08667 w_k1_ADM_s01 - w_k1_ADM_s016
                                 + 0.99 b_k1_ADM_s016 - 1.01 s_k1_ADM_s016  = 0
 w_k1_ADM_s017_return_constraint: 1.50874 w_k1_ADM_s01 - w_k1_ADM_s017
                                 + 0.99 b_k1_ADM_s017 - 1.01 s_k1_ADM_s017  = 0
 w_k1_ADM_s100_return_constraint: 1.08682 w_k1_ADM_s10 - w_k1_ADM_s100
                                 + 0.99 b_k1_ADM_s100 - 1.01 s_k1_ADM_s100  = 0
 w_k1_ADM_s101_return_constraint: 0.911727 w_k1_ADM_s10 - w_k1_ADM_s101
                                 + 0.99 b_k1_ADM_s101 - 1.01 s_k1_ADM_s101  = 0
 w_k1_ADM_s102_return_constraint: 0.768093 w_k1_ADM_s10 - w_k1_ADM_s102
                                 + 0.99 b_k1_ADM_s102 - 1.01 s_k1_ADM_s102  = 0
 w_k1_ADM_s103_return_constraint: 0.912296 w_k1_ADM_s10 - w_k1_ADM_s103
                                 + 0.99 b_k1_ADM_s103 - 1.01 s_k1_ADM_s103  = 0
 w_k1_ADM_s104_return_constraint: 1.12639 w_k1_ADM_s10 - w_k1_ADM_s104
                                 + 0.99 b_k1_ADM_s104 - 1.01 s_k1_ADM_s104  = 0
 w_k1_ADM_s105_return_constraint: 1.04123 w_k1_ADM_s10 - w_k1_ADM_s105
                                 + 0.99 b_k1_ADM_s105 - 1.01 s_k1_ADM_s105  = 0
 w_k1_ADM_s106_return_constraint: 0.898007 w_k1_ADM_s10 - w_k1_ADM_s106
                                 + 0.99 b_k1_ADM_s106 - 1.01 s_k1_ADM_s106  = 0
 w_k1_ADM_s107_return_constraint: 0.747237 w_k1_ADM_s10 - w_k1_ADM_s107
                                 + 0.99 b_k1_ADM_s107 - 1.01 s_k1_ADM_s107  = 0
 w_k1_ADM_s110_return_constraint: 1.21445 w_k1_ADM_s11 - w_k1_ADM_s110
                                 + 0.99 b_k1_ADM_s110 - 1.01 s_k1_ADM_s110  = 0
 w_k1_ADM_s111_return_constraint: 0.748869 w_k1_ADM_s11 - w_k1_ADM_s111
                                 + 0.99 b_k1_ADM_s111 - 1.01 s_k1_ADM_s111  = 0
 w_k1_ADM_s112_return_constraint: 1.05147 w_k1_ADM_s11 - w_k1_ADM_s112
                                 + 0.99 b_k1_ADM_s112 - 1.01 s_k1_ADM_s112  = 0
 w_k1_ADM_s113_return_constraint: 1.14556 w_k1_ADM_s11 - w_k1_ADM_s113
                                 + 0.99 b_k1_ADM_s113 - 1.01 s_k1_ADM_s113  = 0
 w_k1_ADM_s114_return_constraint: 1.29119 w_k1_ADM_s11 - w_k1_ADM_s114
                                 + 0.99 b_k1_ADM_s114 - 1.01 s_k1_ADM_s114  = 0
 w_k1_ADM_s115_return_constraint: 0.889316 w_k1_ADM_s11 - w_k1_ADM_s115
                                 + 0.99 b_k1_ADM_s115 - 1.01 s_k1_ADM_s115  = 0
 w_k1_ADM_s116_return_constraint: 0.652951 w_k1_ADM_s11 - w_k1_ADM_s116
                                 + 0.99 b_k1_ADM_s116 - 1.01 s_k1_ADM_s116  = 0
 w_k1_ADM_s117_return_constraint: 0.916538 w_k1_ADM_s11 - w_k1_ADM_s117
                                 + 0.99 b_k1_ADM_s117 - 1.01 s_k1_ADM_s117  = 0
 w_k1_AXP_s0_return_constraint:  1.21434 w_0_AXP - w_k1_AXP_s0
                                 + 0.99 b_k1_AXP_s0 - 1.01 s_k1_AXP_s0  = 0
 w_k1_AXP_s1_return_constraint:  0.843709 w_0_AXP - w_k1_AXP_s1
                                 + 0.99 b_k1_AXP_s1 - 1.01 s_k1_AXP_s1  = 0
 w_k1_AXP_s00_return_constraint: 1.19829 w_k1_AXP_s0 - w_k1_AXP_s00
                                 + 0.99 b_k1_AXP_s00 - 1.01 s_k1_AXP_s00  = 0
 w_k1_AXP_s01_return_constraint: 0.899753 w_k1_AXP_s0 - w_k1_AXP_s01
                                 + 0.99 b_k1_AXP_s01 - 1.01 s_k1_AXP_s01  = 0
 w_k1_AXP_s10_return_constraint: 1.38453 w_k1_AXP_s1 - w_k1_AXP_s10
                                 + 0.99 b_k1_AXP_s10 - 1.01 s_k1_AXP_s10  = 0
 w_k1_AXP_s11_return_constraint: 0.994972 w_k1_AXP_s1 - w_k1_AXP_s11
                                 + 0.99 b_k1_AXP_s11 - 1.01 s_k1_AXP_s11  = 0
 w_k1_AXP_s000_return_constraint: 0.769741 w_k1_AXP_s00 - w_k1_AXP_s000
                                 + 0.99 b_k1_AXP_s000 - 1.01 s_k1_AXP_s000  = 0
 w_k1_AXP_s001_return_constraint: 1.09753 w_k1_AXP_s00 - w_k1_AXP_s001
                                 + 0.99 b_k1_AXP_s001 - 1.01 s_k1_AXP_s001  = 0
 w_k1_AXP_s002_return_constraint: 0.679998 w_k1_AXP_s00 - w_k1_AXP_s002
                                 + 0.99 b_k1_AXP_s002 - 1.01 s_k1_AXP_s002  = 0
 w_k1_AXP_s003_return_constraint: 1.00542 w_k1_AXP_s00 - w_k1_AXP_s003
                                 + 0.99 b_k1_AXP_s003 - 1.01 s_k1_AXP_s003  = 0
 w_k1_AXP_s004_return_constraint: 1.55889 w_k1_AXP_s00 - w_k1_AXP_s004
                                 + 0.99 b_k1_AXP_s004 - 1.01 s_k1_AXP_s004  = 0
 w_k1_AXP_s005_return_constraint: 1.03243 w_k1_AXP_s00 - w_k1_AXP_s005
                                 + 0.99 b_k1_AXP_s005 - 1.01 s_k1_AXP_s005  = 0
 w_k1_AXP_s006_return_constraint: 1.28751 w_k1_AXP_s00 - w_k1_AXP_s006
                                 + 0.99 b_k1_AXP_s006 - 1.01 s_k1_AXP_s006  = 0
 w_k1_AXP_s007_return_constraint: 0.739499 w_k1_AXP_s00 - w_k1_AXP_s007
                                 + 0.99 b_k1_AXP_s007 - 1.01 s_k1_AXP_s007  = 0
 w_k1_AXP_s010_return_constraint: 1.43591 w_k1_AXP_s01 - w_k1_AXP_s010
                                 + 0.99 b_k1_AXP_s010 - 1.01 s_k1_AXP_s010  = 0
 w_k1_AXP_s011_return_constraint: 0.632752 w_k1_AXP_s01 - w_k1_AXP_s011
                                 + 0.99 b_k1_AXP_s011 - 1.01 s_k1_AXP_s011  = 0
 w_k1_AXP_s012_return_constraint: 1.20216 w_k1_AXP_s01 - w_k1_AXP_s012
                                 + 0.99 b_k1_AXP_s012 - 1.01 s_k1_AXP_s012  = 0
 w_k1_AXP_s013_return_constraint: 0.929089 w_k1_AXP_s01 - w_k1_AXP_s013
                                 + 0.99 b_k1_AXP_s013 - 1.01 s_k1_AXP_s013  = 0
 w_k1_AXP_s014_return_constraint: 0.7844 w_k1_AXP_s01 - w_k1_AXP_s014
                                 + 0.99 b_k1_AXP_s014 - 1.01 s_k1_AXP_s014  = 0
 w_k1_AXP_s015_return_constraint: 1.02477 w_k1_AXP_s01 - w_k1_AXP_s015
                                 + 0.99 b_k1_AXP_s015 - 1.01 s_k1_AXP_s015  = 0
 w_k1_AXP_s016_return_constraint: 1.08542 w_k1_AXP_s01 - w_k1_AXP_s016
                                 + 0.99 b_k1_AXP_s016 - 1.01 s_k1_AXP_s016  = 0
 w_k1_AXP_s017_return_constraint: 1.435 w_k1_AXP_s01 - w_k1_AXP_s017
                                 + 0.99 b_k1_AXP_s017 - 1.01 s_k1_AXP_s017  = 0
 w_k1_AXP_s100_return_constraint: 0.934507 w_k1_AXP_s10 - w_k1_AXP_s100
                                 + 0.99 b_k1_AXP_s100 - 1.01 s_k1_AXP_s100  = 0
 w_k1_AXP_s101_return_constraint: 1.29665 w_k1_AXP_s10 - w_k1_AXP_s101
                                 + 0.99 b_k1_AXP_s101 - 1.01 s_k1_AXP_s101  = 0
 w_k1_AXP_s102_return_constraint: 1.07744 w_k1_AXP_s10 - w_k1_AXP_s102
                                 + 0.99 b_k1_AXP_s102 - 1.01 s_k1_AXP_s102  = 0
 w_k1_AXP_s103_return_constraint: 0.818051 w_k1_AXP_s10 - w_k1_AXP_s103
                                 + 0.99 b_k1_AXP_s103 - 1.01 s_k1_AXP_s103  = 0
 w_k1_AXP_s104_return_constraint: 1.58858 w_k1_AXP_s10 - w_k1_AXP_s104
                                 + 0.99 b_k1_AXP_s104 - 1.01 s_k1_AXP_s104  = 0
 w_k1_AXP_s105_return_constraint: 1.11814 w_k1_AXP_s10 - w_k1_AXP_s105
                                 + 0.99 b_k1_AXP_s105 - 1.01 s_k1_AXP_s105  = 0
 w_k1_AXP_s106_return_constraint: 0.948489 w_k1_AXP_s10 - w_k1_AXP_s106
                                 + 0.99 b_k1_AXP_s106 - 1.01 s_k1_AXP_s106  = 0
 w_k1_AXP_s107_return_constraint: 0.814919 w_k1_AXP_s10 - w_k1_AXP_s107
                                 + 0.99 b_k1_AXP_s107 - 1.01 s_k1_AXP_s107  = 0
 w_k1_AXP_s110_return_constraint: 1.02895 w_k1_AXP_s11 - w_k1_AXP_s110
                                 + 0.99 b_k1_AXP_s110 - 1.01 s_k1_AXP_s110  = 0
 w_k1_AXP_s111_return_constraint: 0.670291 w_k1_AXP_s11 - w_k1_AXP_s111
                                 + 0.99 b_k1_AXP_s111 - 1.01 s_k1_AXP_s111  = 0
 w_k1_AXP_s112_return_constraint: 0.88192 w_k1_AXP_s11 - w_k1_AXP_s112
                                 + 0.99 b_k1_AXP_s112 - 1.01 s_k1_AXP_s112  = 0
 w_k1_AXP_s113_return_constraint: 1.30684 w_k1_AXP_s11 - w_k1_AXP_s113
                                 + 0.99 b_k1_AXP_s113 - 1.01 s_k1_AXP_s113  = 0
 w_k1_AXP_s114_return_constraint: 1.27803 w_k1_AXP_s11 - w_k1_AXP_s114
                                 + 0.99 b_k1_AXP_s114 - 1.01 s_k1_AXP_s114  = 0
 w_k1_AXP_s115_return_constraint: 0.762639 w_k1_AXP_s11 - w_k1_AXP_s115
                                 + 0.99 b_k1_AXP_s115 - 1.01 s_k1_AXP_s115  = 0
 w_k1_AXP_s116_return_constraint: 0.525589 w_k1_AXP_s11 - w_k1_AXP_s116
                                 + 0.99 b_k1_AXP_s116 - 1.01 s_k1_AXP_s116  = 0
 w_k1_AXP_s117_return_constraint: 0.885813 w_k1_AXP_s11 - w_k1_AXP_s117
                                 + 0.99 b_k1_AXP_s117 - 1.01 s_k1_AXP_s117  = 0
 w_k1_DIS_s0_return_constraint:  1.29992 w_0_DIS - w_k1_DIS_s0
                                 + 0.99 b_k1_DIS_s0 - 1.01 s_k1_DIS_s0  = 0
 w_k1_DIS_s1_return_constraint:  0.809544 w_0_DIS - w_k1_DIS_s1
                                 + 0.99 b_k1_DIS_s1 - 1.01 s_k1_DIS_s1  = 0
 w_k1_DIS_s00_return_constraint: 1.34539 w_k1_DIS_s0 - w_k1_DIS_s00
                                 + 0.99 b_k1_DIS_s00 - 1.01 s_k1_DIS_s00  = 0
 w_k1_DIS_s01_return_constraint: 0.830693 w_k1_DIS_s0 - w_k1_DIS_s01
                                 + 0.99 b_k1_DIS_s01 - 1.01 s_k1_DIS_s01  = 0
 w_k1_DIS_s10_return_constraint: 1.54803 w_k1_DIS_s1 - w_k1_DIS_s10
                                 + 0.99 b_k1_DIS_s10 - 1.01 s_k1_DIS_s10  = 0
 w_k1_DIS_s11_return_constraint: 0.942514 w_k1_DIS_s1 - w_k1_DIS_s11
                                 + 0.99 b_k1_DIS_s11 - 1.01 s_k1_DIS_s11  = 0
 w_k1_DIS_s000_return_constraint: 0.725431 w_k1_DIS_s00 - w_k1_DIS_s000
                                 + 0.99 b_k1_DIS_s000 - 1.01 s_k1_DIS_s000  = 0
 w_k1_DIS_s001_return_constraint: 1.15534 w_k1_DIS_s00 - w_k1_DIS_s001
                                 + 0.99 b_k1_DIS_s001 - 1.01 s_k1_DIS_s001  = 0
 w_k1_DIS_s002_return_constraint: 0.564667 w_k1_DIS_s00 - w_k1_DIS_s002
                                 + 0.99 b_k1_DIS_s002 - 1.01 s_k1_DIS_s002  = 0
 w_k1_DIS_s003_return_constraint: 0.971082 w_k1_DIS_s00 - w_k1_DIS_s003
                                 + 0.99 b_k1_DIS_s003 - 1.01 s_k1_DIS_s003  = 0
 w_k1_DIS_s004_return_constraint: 1.71884 w_k1_DIS_s00 - w_k1_DIS_s004
                                 + 0.99 b_k1_DIS_s004 - 1.01 s_k1_DIS_s004  = 0
 w_k1_DIS_s005_return_constraint: 0.86856 w_k1_DIS_s00 - w_k1_DIS_s005
                                 + 0.99 b_k1_DIS_s005 - 1.01 s_k1_DIS_s005  = 0
 w_k1_DIS_s006_return_constraint: 1.09286 w_k1_DIS_s00 - w_k1_DIS_s006
                                 + 0.99 b_k1_DIS_s006 - 1.01 s_k1_DIS_s006  = 0
 w_k1_DIS_s007_return_constraint: 0.867156 w_k1_DIS_s00 - w_k1_DIS_s007
                                 + 0.99 b_k1_DIS_s007 - 1.01 s_k1_DIS_s007  = 0
 w_k1_DIS_s010_return_constraint: 1.30167 w_k1_DIS_s01 - w_k1_DIS_s010
                                 + 0.99 b_k1_DIS_s010 - 1.01 s_k1_DIS_s010  = 0
 w_k1_DIS_s011_return_constraint: 0.72278 w_k1_DIS_s01 - w_k1_DIS_s011
                                 + 0.99 b_k1_DIS_s011 - 1.01 s_k1_DIS_s011  = 0
 w_k1_DIS_s012_return_constraint: 1.13961 w_k1_DIS_s01 - w_k1_DIS_s012
                                 + 0.99 b_k1_DIS_s012 - 1.01 s_k1_DIS_s012  = 0
 w_k1_DIS_s013_return_constraint: 0.807256 w_k1_DIS_s01 - w_k1_DIS_s013
                                 + 0.99 b_k1_DIS_s013 - 1.01 s_k1_DIS_s013  = 0
 w_k1_DIS_s014_return_constraint: 0.6785 w_k1_DIS_s01 - w_k1_DIS_s014
                                 + 0.99 b_k1_DIS_s014 - 1.01 s_k1_DIS_s014  = 0
 w_k1_DIS_s015_return_constraint: 1.19675 w_k1_DIS_s01 - w_k1_DIS_s015
                                 + 0.99 b_k1_DIS_s015 - 1.01 s_k1_DIS_s015  = 0
 w_k1_DIS_s016_return_constraint: 1.08895 w_k1_DIS_s01 - w_k1_DIS_s016
                                 + 0.99 b_k1_DIS_s016 - 1.01 s_k1_DIS_s016  = 0
 w_k1_DIS_s017_return_constraint: 1.7826 w_k1_DIS_s01 - w_k1_DIS_s017
                                 + 0.99 b_k1_DIS_s017 - 1.01 s_k1_DIS_s017  = 0
 w_k1_DIS_s100_return_constraint: 0.929715 w_k1_DIS_s10 - w_k1_DIS_s100
                                 + 0.99 b_k1_DIS_s100 - 1.01 s_k1_DIS_s100  = 0
 w_k1_DIS_s101_return_constraint: 1.09527 w_k1_DIS_s10 - w_k1_DIS_s101
                                 + 0.99 b_k1_DIS_s101 - 1.01 s_k1_DIS_s101  = 0
 w_k1_DIS_s102_return_constraint: 1.09502 w_k1_DIS_s10 - w_k1_DIS_s102
                                 + 0.99 b_k1_DIS_s102 - 1.01 s_k1_DIS_s102  = 0
 w_k1_DIS_s103_return_constraint: 0.903691 w_k1_DIS_s10 - w_k1_DIS_s103
                                 + 0.99 b_k1_DIS_s103 - 1.01 s_k1_DIS_s103  = 0
 w_k1_DIS_s104_return_constraint: 1.71262 w_k1_DIS_s10 - w_k1_DIS_s104
                                 + 0.99 b_k1_DIS_s104 - 1.01 s_k1_DIS_s104  = 0
 w_k1_DIS_s105_return_constraint: 1.44552 w_k1_DIS_s10 - w_k1_DIS_s105
                                 + 0.99 b_k1_DIS_s105 - 1.01 s_k1_DIS_s105  = 0
 w_k1_DIS_s106_return_constraint: 0.784683 w_k1_DIS_s10 - w_k1_DIS_s106
                                 + 0.99 b_k1_DIS_s106 - 1.01 s_k1_DIS_s106  = 0
 w_k1_DIS_s107_return_constraint: 0.743995 w_k1_DIS_s10 - w_k1_DIS_s107
                                 + 0.99 b_k1_DIS_s107 - 1.01 s_k1_DIS_s107  = 0
 w_k1_DIS_s110_return_constraint: 1.14093 w_k1_DIS_s11 - w_k1_DIS_s110
                                 + 0.99 b_k1_DIS_s110 - 1.01 s_k1_DIS_s110  = 0
 w_k1_DIS_s111_return_constraint: 0.574964 w_k1_DIS_s11 - w_k1_DIS_s111
                                 + 0.99 b_k1_DIS_s111 - 1.01 s_k1_DIS_s111  = 0
 w_k1_DIS_s112_return_constraint: 0.834542 w_k1_DIS_s11 - w_k1_DIS_s112
                                 + 0.99 b_k1_DIS_s112 - 1.01 s_k1_DIS_s112  = 0
 w_k1_DIS_s113_return_constraint: 1.23445 w_k1_DIS_s11 - w_k1_DIS_s113
                                 + 0.99 b_k1_DIS_s113 - 1.01 s_k1_DIS_s113  = 0
 w_k1_DIS_s114_return_constraint: 1.39949 w_k1_DIS_s11 - w_k1_DIS_s114
                                 + 0.99 b_k1_DIS_s114 - 1.01 s_k1_DIS_s114  = 0
 w_k1_DIS_s115_return_constraint: 0.675932 w_k1_DIS_s11 - w_k1_DIS_s115
                                 + 0.99 b_k1_DIS_s115 - 1.01 s_k1_DIS_s115  = 0
 w_k1_DIS_s116_return_constraint: 0.490712 w_k1_DIS_s11 - w_k1_DIS_s116
                                 + 0.99 b_k1_DIS_s116 - 1.01 s_k1_DIS_s116  = 0
 w_k1_DIS_s117_return_constraint: 1.01842 w_k1_DIS_s11 - w_k1_DIS_s117
                                 + 0.99 b_k1_DIS_s117 - 1.01 s_k1_DIS_s117  = 0
 w_k1_FDX_s0_return_constraint:  1.15921 w_0_FDX - w_k1_FDX_s0
                                 + 0.99 b_k1_FDX_s0 - 1.01 s_k1_FDX_s0  = 0
 w_k1_FDX_s1_return_constraint:  0.891811 w_0_FDX - w_k1_FDX_s1
                                 + 0.99 b_k1_FDX_s1 - 1.01 s_k1_FDX_s1  = 0
 w_k1_FDX_s00_return_constraint: 1.15876 w_k1_FDX_s0 - w_k1_FDX_s00
                                 + 0.99 b_k1_FDX_s00 - 1.01 s_k1_FDX_s00  = 0
 w_k1_FDX_s01_return_constraint: 0.926529 w_k1_FDX_s0 - w_k1_FDX_s01
                                 + 0.99 b_k1_FDX_s01 - 1.01 s_k1_FDX_s01  = 0
 w_k1_FDX_s10_return_constraint: 1.31654 w_k1_FDX_s1 - w_k1_FDX_s10
                                 + 0.99 b_k1_FDX_s10 - 1.01 s_k1_FDX_s10  = 0
 w_k1_FDX_s11_return_constraint: 0.93144 w_k1_FDX_s1 - w_k1_FDX_s11
                                 + 0.99 b_k1_FDX_s11 - 1.01 s_k1_FDX_s11  = 0
 w_k1_FDX_s000_return_constraint: 0.762495 w_k1_FDX_s00 - w_k1_FDX_s000
                                 + 0.99 b_k1_FDX_s000 - 1.01 s_k1_FDX_s000  = 0
 w_k1_FDX_s001_return_constraint: 1.01502 w_k1_FDX_s00 - w_k1_FDX_s001
                                 + 0.99 b_k1_FDX_s001 - 1.01 s_k1_FDX_s001  = 0
 w_k1_FDX_s002_return_constraint: 0.658133 w_k1_FDX_s00 - w_k1_FDX_s002
                                 + 0.99 b_k1_FDX_s002 - 1.01 s_k1_FDX_s002  = 0
 w_k1_FDX_s003_return_constraint: 1.10722 w_k1_FDX_s00 - w_k1_FDX_s003
                                 + 0.99 b_k1_FDX_s003 - 1.01 s_k1_FDX_s003  = 0
 w_k1_FDX_s004_return_constraint: 1.50072 w_k1_FDX_s00 - w_k1_FDX_s004
                                 + 0.99 b_k1_FDX_s004 - 1.01 s_k1_FDX_s004  = 0
 w_k1_FDX_s005_return_constraint: 0.905564 w_k1_FDX_s00 - w_k1_FDX_s005
                                 + 0.99 b_k1_FDX_s005 - 1.01 s_k1_FDX_s005  = 0
 w_k1_FDX_s006_return_constraint: 0.984882 w_k1_FDX_s00 - w_k1_FDX_s006
                                 + 0.99 b_k1_FDX_s006 - 1.01 s_k1_FDX_s006  = 0
 w_k1_FDX_s007_return_constraint: 0.914828 w_k1_FDX_s00 - w_k1_FDX_s007
                                 + 0.99 b_k1_FDX_s007 - 1.01 s_k1_FDX_s007  = 0
 w_k1_FDX_s010_return_constraint: 1.23866 w_k1_FDX_s01 - w_k1_FDX_s010
                                 + 0.99 b_k1_FDX_s010 - 1.01 s_k1_FDX_s010  = 0
 w_k1_FDX_s011_return_constraint: 0.749982 w_k1_FDX_s01 - w_k1_FDX_s011
                                 + 0.99 b_k1_FDX_s011 - 1.01 s_k1_FDX_s011  = 0
 w_k1_FDX_s012_return_constraint: 0.935448 w_k1_FDX_s01 - w_k1_FDX_s012
                                 + 0.99 b_k1_FDX_s012 - 1.01 s_k1_FDX_s012  = 0
 w_k1_FDX_s013_return_constraint: 0.84086 w_k1_FDX_s01 - w_k1_FDX_s013
                                 + 0.99 b_k1_FDX_s013 - 1.01 s_k1_FDX_s013  = 0
 w_k1_FDX_s014_return_constraint: 0.848307 w_k1_FDX_s01 - w_k1_FDX_s014
                                 + 0.99 b_k1_FDX_s014 - 1.01 s_k1_FDX_s014  = 0
 w_k1_FDX_s015_return_constraint: 1.06669 w_k1_FDX_s01 - w_k1_FDX_s015
                                 + 0.99 b_k1_FDX_s015 - 1.01 s_k1_FDX_s015  = 0
 w_k1_FDX_s016_return_constraint: 1.12237 w_k1_FDX_s01 - w_k1_FDX_s016
                                 + 0.99 b_k1_FDX_s016 - 1.01 s_k1_FDX_s016  = 0
 w_k1_FDX_s017_return_constraint: 1.53385 w_k1_FDX_s01 - w_k1_FDX_s017
                                 + 0.99 b_k1_FDX_s017 - 1.01 s_k1_FDX_s017  = 0
 w_k1_FDX_s100_return_constraint: 1.06354 w_k1_FDX_s10 - w_k1_FDX_s100
                                 + 0.99 b_k1_FDX_s100 - 1.01 s_k1_FDX_s100  = 0
 w_k1_FDX_s101_return_constraint: 1.11426 w_k1_FDX_s10 - w_k1_FDX_s101
                                 + 0.99 b_k1_FDX_s101 - 1.01 s_k1_FDX_s101  = 0
 w_k1_FDX_s102_return_constraint: 0.81124 w_k1_FDX_s10 - w_k1_FDX_s102
                                 + 0.99 b_k1_FDX_s102 - 1.01 s_k1_FDX_s102  = 0
 w_k1_FDX_s103_return_constraint: 0.889939 w_k1_FDX_s10 - w_k1_FDX_s103
                                 + 0.99 b_k1_FDX_s103 - 1.01 s_k1_FDX_s103  = 0
 w_k1_FDX_s104_return_constraint: 1.3621 w_k1_FDX_s10 - w_k1_FDX_s104
                                 + 0.99 b_k1_FDX_s104 - 1.01 s_k1_FDX_s104  = 0
 w_k1_FDX_s105_return_constraint: 1.09567 w_k1_FDX_s10 - w_k1_FDX_s105
                                 + 0.99 b_k1_FDX_s105 - 1.01 s_k1_FDX_s105  = 0
 w_k1_FDX_s106_return_constraint: 0.802924 w_k1_FDX_s10 - w_k1_FDX_s106
                                 + 0.99 b_k1_FDX_s106 - 1.01 s_k1_FDX_s106  = 0
 w_k1_FDX_s107_return_constraint: 0.817179 w_k1_FDX_s10 - w_k1_FDX_s107
                                 + 0.99 b_k1_FDX_s107 - 1.01 s_k1_FDX_s107  = 0
 w_k1_FDX_s110_return_constraint: 1.24129 w_k1_FDX_s11 - w_k1_FDX_s110
                                 + 0.99 b_k1_FDX_s110 - 1.01 s_k1_FDX_s110  = 0
 w_k1_FDX_s111_return_constraint: 0.632678 w_k1_FDX_s11 - w_k1_FDX_s111
                                 + 0.99 b_k1_FDX_s111 - 1.01 s_k1_FDX_s111  = 0
 w_k1_FDX_s112_return_constraint: 0.867218 w_k1_FDX_s11 - w_k1_FDX_s112
                                 + 0.99 b_k1_FDX_s112 - 1.01 s_k1_FDX_s112  = 0
 w_k1_FDX_s113_return_constraint: 1.09222 w_k1_FDX_s11 - w_k1_FDX_s113
                                 + 0.99 b_k1_FDX_s113 - 1.01 s_k1_FDX_s113  = 0
 w_k1_FDX_s114_return_constraint: 1.31119 w_k1_FDX_s11 - w_k1_FDX_s114
                                 + 0.99 b_k1_FDX_s114 - 1.01 s_k1_FDX_s114  = 0
 w_k1_FDX_s115_return_constraint: 0.877822 w_k1_FDX_s11 - w_k1_FDX_s115
                                 + 0.99 b_k1_FDX_s115 - 1.01 s_k1_FDX_s115  = 0
 w_k1_FDX_s116_return_constraint: 0.713687 w_k1_FDX_s11 - w_k1_FDX_s116
                                 + 0.99 b_k1_FDX_s116 - 1.01 s_k1_FDX_s116  = 0
 w_k1_FDX_s117_return_constraint: 1.00075 w_k1_FDX_s11 - w_k1_FDX_s117
                                 + 0.99 b_k1_FDX_s117 - 1.01 s_k1_FDX_s117  = 0
 w_k1_IBM_s0_return_constraint:  1.14143 w_0_IBM - w_k1_IBM_s0
                                 + 0.99 b_k1_IBM_s0 - 1.01 s_k1_IBM_s0  = 0
 w_k1_IBM_s1_return_constraint:  0.904496 w_0_IBM - w_k1_IBM_s1
                                 + 0.99 b_k1_IBM_s1 - 1.01 s_k1_IBM_s1  = 0
 w_k1_IBM_s00_return_constraint: 1.36821 w_k1_IBM_s0 - w_k1_IBM_s00
                                 + 0.99 b_k1_IBM_s00 - 1.01 s_k1_IBM_s00  = 0
 w_k1_IBM_s01_return_constraint: 0.892175 w_k1_IBM_s0 - w_k1_IBM_s01
                                 + 0.99 b_k1_IBM_s01 - 1.01 s_k1_IBM_s01  = 0
 w_k1_IBM_s10_return_constraint: 1.26672 w_k1_IBM_s1 - w_k1_IBM_s10
                                 + 0.99 b_k1_IBM_s10 - 1.01 s_k1_IBM_s10  = 0
 w_k1_IBM_s11_return_constraint: 0.992354 w_k1_IBM_s1 - w_k1_IBM_s11
                                 + 0.99 b_k1_IBM_s11 - 1.01 s_k1_IBM_s11  = 0
 w_k1_IBM_s000_return_constraint: 0.966644 w_k1_IBM_s00 - w_k1_IBM_s000
                                 + 0.99 b_k1_IBM_s000 - 1.01 s_k1_IBM_s000  = 0
 w_k1_IBM_s001_return_constraint: 1.21808 w_k1_IBM_s00 - w_k1_IBM_s001
                                 + 0.99 b_k1_IBM_s001 - 1.01 s_k1_IBM_s001  = 0
 w_k1_IBM_s002_return_constraint: 0.932392 w_k1_IBM_s00 - w_k1_IBM_s002
                                 + 0.99 b_k1_IBM_s002 - 1.01 s_k1_IBM_s002  = 0
 w_k1_IBM_s003_return_constraint: 0.752848 w_k1_IBM_s00 - w_k1_IBM_s003
                                 + 0.99 b_k1_IBM_s003 - 1.01 s_k1_IBM_s003  = 0
 w_k1_IBM_s004_return_constraint: 0.965073 w_k1_IBM_s00 - w_k1_IBM_s004
                                 + 0.99 b_k1_IBM_s004 - 1.01 s_k1_IBM_s004  = 0
 w_k1_IBM_s005_return_constraint: 1.01388 w_k1_IBM_s00 - w_k1_IBM_s005
                                 + 0.99 b_k1_IBM_s005 - 1.01 s_k1_IBM_s005  = 0
 w_k1_IBM_s006_return_constraint: 1.83367 w_k1_IBM_s00 - w_k1_IBM_s006
                                 + 0.99 b_k1_IBM_s006 - 1.01 s_k1_IBM_s006  = 0
 w_k1_IBM_s007_return_constraint: 0.62558 w_k1_IBM_s00 - w_k1_IBM_s007
                                 + 0.99 b_k1_IBM_s007 - 1.01 s_k1_IBM_s007  = 0
 w_k1_IBM_s010_return_constraint: 1.10516 w_k1_IBM_s01 - w_k1_IBM_s010
                                 + 0.99 b_k1_IBM_s010 - 1.01 s_k1_IBM_s010  = 0
 w_k1_IBM_s011_return_constraint: 0.859423 w_k1_IBM_s01 - w_k1_IBM_s011
                                 + 0.99 b_k1_IBM_s011 - 1.01 s_k1_IBM_s011  = 0
 w_k1_IBM_s012_return_constraint: 1.5974 w_k1_IBM_s01 - w_k1_IBM_s012
                                 + 0.99 b_k1_IBM_s012 - 1.01 s_k1_IBM_s012  = 0
 w_k1_IBM_s013_return_constraint: 1.08855 w_k1_IBM_s01 - w_k1_IBM_s013
                                 + 0.99 b_k1_IBM_s013 - 1.01 s_k1_IBM_s013  = 0
 w_k1_IBM_s014_return_constraint: 0.540965 w_k1_IBM_s01 - w_k1_IBM_s014
                                 + 0.99 b_k1_IBM_s014 - 1.01 s_k1_IBM_s014  = 0
 w_k1_IBM_s015_return_constraint: 1.15758 w_k1_IBM_s01 - w_k1_IBM_s015
                                 + 0.99 b_k1_IBM_s015 - 1.01 s_k1_IBM_s015  = 0
 w_k1_IBM_s016_return_constraint: 0.710095 w_k1_IBM_s01 - w_k1_IBM_s016
                                 + 0.99 b_k1_IBM_s016 - 1.01 s_k1_IBM_s016  = 0
 w_k1_IBM_s017_return_constraint: 0.997295 w_k1_IBM_s01 - w_k1_IBM_s017
                                 + 0.99 b_k1_IBM_s017 - 1.01 s_k1_IBM_s017  = 0
 w_k1_IBM_s100_return_constraint: 0.56722 w_k1_IBM_s10 - w_k1_IBM_s100
                                 + 0.99 b_k1_IBM_s100 - 1.01 s_k1_IBM_s100  = 0
 w_k1_IBM_s101_return_constraint: 1.09393 w_k1_IBM_s10 - w_k1_IBM_s101
                                 + 0.99 b_k1_IBM_s101 - 1.01 s_k1_IBM_s101  = 0
 w_k1_IBM_s102_return_constraint: 1.73632 w_k1_IBM_s10 - w_k1_IBM_s102
                                 + 0.99 b_k1_IBM_s102 - 1.01 s_k1_IBM_s102  = 0
 w_k1_IBM_s103_return_constraint: 1.23604 w_k1_IBM_s10 - w_k1_IBM_s103
                                 + 0.99 b_k1_IBM_s103 - 1.01 s_k1_IBM_s103  = 0
 w_k1_IBM_s104_return_constraint: 1.09569 w_k1_IBM_s10 - w_k1_IBM_s104
                                 + 0.99 b_k1_IBM_s104 - 1.01 s_k1_IBM_s104  = 0
 w_k1_IBM_s105_return_constraint: 1.43301 w_k1_IBM_s10 - w_k1_IBM_s105
                                 + 0.99 b_k1_IBM_s105 - 1.01 s_k1_IBM_s105  = 0
 w_k1_IBM_s106_return_constraint: 1.05095 w_k1_IBM_s10 - w_k1_IBM_s106
                                 + 0.99 b_k1_IBM_s106 - 1.01 s_k1_IBM_s106  = 0
 w_k1_IBM_s107_return_constraint: 0.779533 w_k1_IBM_s10 - w_k1_IBM_s107
                                 + 0.99 b_k1_IBM_s107 - 1.01 s_k1_IBM_s107  = 0
 w_k1_IBM_s110_return_constraint: 0.696731 w_k1_IBM_s11 - w_k1_IBM_s110
                                 + 0.99 b_k1_IBM_s110 - 1.01 s_k1_IBM_s110  = 0
 w_k1_IBM_s111_return_constraint: 0.930939 w_k1_IBM_s11 - w_k1_IBM_s111
                                 + 0.99 b_k1_IBM_s111 - 1.01 s_k1_IBM_s111  = 0
 w_k1_IBM_s112_return_constraint: 1.04389 w_k1_IBM_s11 - w_k1_IBM_s112
                                 + 0.99 b_k1_IBM_s112 - 1.01 s_k1_IBM_s112  = 0
 w_k1_IBM_s113_return_constraint: 1.94939 w_k1_IBM_s11 - w_k1_IBM_s113
                                 + 0.99 b_k1_IBM_s113 - 1.01 s_k1_IBM_s113  = 0
 w_k1_IBM_s114_return_constraint: 1.2369 w_k1_IBM_s11 - w_k1_IBM_s114
                                 + 0.99 b_k1_IBM_s114 - 1.01 s_k1_IBM_s114  = 0
 w_k1_IBM_s115_return_constraint: 0.652434 w_k1_IBM_s11 - w_k1_IBM_s115
                                 + 0.99 b_k1_IBM_s115 - 1.01 s_k1_IBM_s115  = 0
 w_k1_IBM_s116_return_constraint: 0.538928 w_k1_IBM_s11 - w_k1_IBM_s116
                                 + 0.99 b_k1_IBM_s116 - 1.01 s_k1_IBM_s116  = 0
 w_k1_IBM_s117_return_constraint: 1.06244 w_k1_IBM_s11 - w_k1_IBM_s117
                                 + 0.99 b_k1_IBM_s117 - 1.01 s_k1_IBM_s117  = 0
 w_k1_INTC_s0_return_constraint: 1.13022 w_0_INTC - w_k1_INTC_s0
                                 + 0.99 b_k1_INTC_s0 - 1.01 s_k1_INTC_s0  = 0
 w_k1_INTC_s1_return_constraint: 0.865002 w_0_INTC - w_k1_INTC_s1
                                 + 0.99 b_k1_INTC_s1 - 1.01 s_k1_INTC_s1  = 0
 w_k1_INTC_s00_return_constraint: 1.15892 w_k1_INTC_s0 - w_k1_INTC_s00
                                 + 0.99 b_k1_INTC_s00 - 1.01 s_k1_INTC_s00  = 0
 w_k1_INTC_s01_return_constraint: 0.940052 w_k1_INTC_s0 - w_k1_INTC_s01
                                 + 0.99 b_k1_INTC_s01 - 1.01 s_k1_INTC_s01  = 0
 w_k1_INTC_s10_return_constraint: 1.30587 w_k1_INTC_s1 - w_k1_INTC_s10
                                 + 0.99 b_k1_INTC_s10 - 1.01 s_k1_INTC_s10  = 0
 w_k1_INTC_s11_return_constraint: 1.00296 w_k1_INTC_s1 - w_k1_INTC_s11
                                 + 0.99 b_k1_INTC_s11 - 1.01 s_k1_INTC_s11  = 0
 w_k1_INTC_s000_return_constraint: 0.779284 w_k1_INTC_s00 - w_k1_INTC_s000
                                 + 0.99 b_k1_INTC_s000 - 1.01 s_k1_INTC_s000
                                  = 0
 w_k1_INTC_s001_return_constraint: 1.27364 w_k1_INTC_s00 - w_k1_INTC_s001
                                 + 0.99 b_k1_INTC_s001 - 1.01 s_k1_INTC_s001
                                  = 0
 w_k1_INTC_s002_return_constraint: 0.760438 w_k1_INTC_s00 - w_k1_INTC_s002
                                 + 0.99 b_k1_INTC_s002 - 1.01 s_k1_INTC_s002
                                  = 0
 w_k1_INTC_s003_return_constraint: 1.24048 w_k1_INTC_s00 - w_k1_INTC_s003
                                 + 0.99 b_k1_INTC_s003 - 1.01 s_k1_INTC_s003
                                  = 0
 w_k1_INTC_s004_return_constraint: 1.23308 w_k1_INTC_s00 - w_k1_INTC_s004
                                 + 0.99 b_k1_INTC_s004 - 1.01 s_k1_INTC_s004
                                  = 0
 w_k1_INTC_s005_return_constraint: 0.852066 w_k1_INTC_s00 - w_k1_INTC_s005
                                 + 0.99 b_k1_INTC_s005 - 1.01 s_k1_INTC_s005
                                  = 0
 w_k1_INTC_s006_return_constraint: 0.921156 w_k1_INTC_s00 - w_k1_INTC_s006
                                 + 0.99 b_k1_INTC_s006 - 1.01 s_k1_INTC_s006
                                  = 0
 w_k1_INTC_s007_return_constraint: 0.860729 w_k1_INTC_s00 - w_k1_INTC_s007
                                 + 0.99 b_k1_INTC_s007 - 1.01 s_k1_INTC_s007
                                  = 0
 w_k1_INTC_s010_return_constraint: 1.14268 w_k1_INTC_s01 - w_k1_INTC_s010
                                 + 0.99 b_k1_INTC_s010 - 1.01 s_k1_INTC_s010
                                  = 0
 w_k1_INTC_s011_return_constraint: 0.758081 w_k1_INTC_s01 - w_k1_INTC_s011
                                 + 0.99 b_k1_INTC_s011 - 1.01 s_k1_INTC_s011
                                  = 0
 w_k1_INTC_s012_return_constraint: 0.991781 w_k1_INTC_s01 - w_k1_INTC_s012
                                 + 0.99 b_k1_INTC_s012 - 1.01 s_k1_INTC_s012
                                  = 0
 w_k1_INTC_s013_return_constraint: 0.856964 w_k1_INTC_s01 - w_k1_INTC_s013
                                 + 0.99 b_k1_INTC_s013 - 1.01 s_k1_INTC_s013
                                  = 0
 w_k1_INTC_s014_return_constraint: 0.843489 w_k1_INTC_s01 - w_k1_INTC_s014
                                 + 0.99 b_k1_INTC_s014 - 1.01 s_k1_INTC_s014
                                  = 0
 w_k1_INTC_s015_return_constraint: 0.90499 w_k1_INTC_s01 - w_k1_INTC_s015
                                 + 0.99 b_k1_INTC_s015 - 1.01 s_k1_INTC_s015
                                  = 0
 w_k1_INTC_s016_return_constraint: 0.984682 w_k1_INTC_s01 - w_k1_INTC_s016
                                 + 0.99 b_k1_INTC_s016 - 1.01 s_k1_INTC_s016
                                  = 0
 w_k1_INTC_s017_return_constraint: 1.27372 w_k1_INTC_s01 - w_k1_INTC_s017
                                 + 0.99 b_k1_INTC_s017 - 1.01 s_k1_INTC_s017
                                  = 0
 w_k1_INTC_s100_return_constraint: 0.960067 w_k1_INTC_s10 - w_k1_INTC_s100
                                 + 0.99 b_k1_INTC_s100 - 1.01 s_k1_INTC_s100
                                  = 0
 w_k1_INTC_s101_return_constraint: 1.06557 w_k1_INTC_s10 - w_k1_INTC_s101
                                 + 0.99 b_k1_INTC_s101 - 1.01 s_k1_INTC_s101
                                  = 0
 w_k1_INTC_s102_return_constraint: 0.867303 w_k1_INTC_s10 - w_k1_INTC_s102
                                 + 0.99 b_k1_INTC_s102 - 1.01 s_k1_INTC_s102
                                  = 0
 w_k1_INTC_s103_return_constraint: 0.99397 w_k1_INTC_s10 - w_k1_INTC_s103
                                 + 0.99 b_k1_INTC_s103 - 1.01 s_k1_INTC_s103
                                  = 0
 w_k1_INTC_s104_return_constraint: 1.12119 w_k1_INTC_s10 - w_k1_INTC_s104
                                 + 0.99 b_k1_INTC_s104 - 1.01 s_k1_INTC_s104
                                  = 0
 w_k1_INTC_s105_return_constraint: 1.20252 w_k1_INTC_s10 - w_k1_INTC_s105
                                 + 0.99 b_k1_INTC_s105 - 1.01 s_k1_INTC_s105
                                  = 0
 w_k1_INTC_s106_return_constraint: 0.769816 w_k1_INTC_s10 - w_k1_INTC_s106
                                 + 0.99 b_k1_INTC_s106 - 1.01 s_k1_INTC_s106
                                  = 0
 w_k1_INTC_s107_return_constraint: 0.797243 w_k1_INTC_s10 - w_k1_INTC_s107
                                 + 0.99 b_k1_INTC_s107 - 1.01 s_k1_INTC_s107
                                  = 0
 w_k1_INTC_s110_return_constraint: 1.13184 w_k1_INTC_s11 - w_k1_INTC_s110
                                 + 0.99 b_k1_INTC_s110 - 1.01 s_k1_INTC_s110
                                  = 0
 w_k1_INTC_s111_return_constraint: 0.703809 w_k1_INTC_s11 - w_k1_INTC_s111
                                 + 0.99 b_k1_INTC_s111 - 1.01 s_k1_INTC_s111
                                  = 0
 w_k1_INTC_s112_return_constraint: 0.865084 w_k1_INTC_s11 - w_k1_INTC_s112
                                 + 0.99 b_k1_INTC_s112 - 1.01 s_k1_INTC_s112
                                  = 0
 w_k1_INTC_s113_return_constraint: 0.995153 w_k1_INTC_s11 - w_k1_INTC_s113
                                 + 0.99 b_k1_INTC_s113 - 1.01 s_k1_INTC_s113
                                  = 0
 w_k1_INTC_s114_return_constraint: 1.13819 w_k1_INTC_s11 - w_k1_INTC_s114
                                 + 0.99 b_k1_INTC_s114 - 1.01 s_k1_INTC_s114
                                  = 0
 w_k1_INTC_s115_return_constraint: 0.88521 w_k1_INTC_s11 - w_k1_INTC_s115
                                 + 0.99 b_k1_INTC_s115 - 1.01 s_k1_INTC_s115
                                  = 0
 w_k1_INTC_s116_return_constraint: 0.811726 w_k1_INTC_s11 - w_k1_INTC_s116
                                 + 0.99 b_k1_INTC_s116 - 1.01 s_k1_INTC_s116
                                  = 0
 w_k1_INTC_s117_return_constraint: 1.0275 w_k1_INTC_s11 - w_k1_INTC_s117
                                 + 0.99 b_k1_INTC_s117 - 1.01 s_k1_INTC_s117
                                  = 0
 w_k1_KO_s0_return_constraint:   1.15241 w_0_KO - w_k1_KO_s0 + 0.99 b_k1_KO_s0
                                 - 1.01 s_k1_KO_s0  = 0
 w_k1_KO_s1_return_constraint:   0.884797 w_0_KO - w_k1_KO_s1 + 0.99 b_k1_KO_s1
                                 - 1.01 s_k1_KO_s1  = 0
 w_k1_KO_s00_return_constraint:  1.44554 w_k1_KO_s0 - w_k1_KO_s00
                                 + 0.99 b_k1_KO_s00 - 1.01 s_k1_KO_s00  = 0
 w_k1_KO_s01_return_constraint:  0.916913 w_k1_KO_s0 - w_k1_KO_s01
                                 + 0.99 b_k1_KO_s01 - 1.01 s_k1_KO_s01  = 0
 w_k1_KO_s10_return_constraint:  1.28905 w_k1_KO_s1 - w_k1_KO_s10
                                 + 0.99 b_k1_KO_s10 - 1.01 s_k1_KO_s10  = 0
 w_k1_KO_s11_return_constraint:  0.923242 w_k1_KO_s1 - w_k1_KO_s11
                                 + 0.99 b_k1_KO_s11 - 1.01 s_k1_KO_s11  = 0
 w_k1_KO_s000_return_constraint: 0.772728 w_k1_KO_s00 - w_k1_KO_s000
                                 + 0.99 b_k1_KO_s000 - 1.01 s_k1_KO_s000  = 0
 w_k1_KO_s001_return_constraint: 1.09163 w_k1_KO_s00 - w_k1_KO_s001
                                 + 0.99 b_k1_KO_s001 - 1.01 s_k1_KO_s001  = 0
 w_k1_KO_s002_return_constraint: 0.673533 w_k1_KO_s00 - w_k1_KO_s002
                                 + 0.99 b_k1_KO_s002 - 1.01 s_k1_KO_s002  = 0
 w_k1_KO_s003_return_constraint: 0.793669 w_k1_KO_s00 - w_k1_KO_s003
                                 + 0.99 b_k1_KO_s003 - 1.01 s_k1_KO_s003  = 0
 w_k1_KO_s004_return_constraint: 1.10864 w_k1_KO_s00 - w_k1_KO_s004
                                 + 0.99 b_k1_KO_s004 - 1.01 s_k1_KO_s004  = 0
 w_k1_KO_s005_return_constraint: 0.75017 w_k1_KO_s00 - w_k1_KO_s005
                                 + 0.99 b_k1_KO_s005 - 1.01 s_k1_KO_s005  = 0
 w_k1_KO_s006_return_constraint: 1.20993 w_k1_KO_s00 - w_k1_KO_s006
                                 + 0.99 b_k1_KO_s006 - 1.01 s_k1_KO_s006  = 0
 w_k1_KO_s007_return_constraint: 0.776799 w_k1_KO_s00 - w_k1_KO_s007
                                 + 0.99 b_k1_KO_s007 - 1.01 s_k1_KO_s007  = 0
 w_k1_KO_s010_return_constraint: 1.15845 w_k1_KO_s01 - w_k1_KO_s010
                                 + 0.99 b_k1_KO_s010 - 1.01 s_k1_KO_s010  = 0
 w_k1_KO_s011_return_constraint: 0.760592 w_k1_KO_s01 - w_k1_KO_s011
                                 + 0.99 b_k1_KO_s011 - 1.01 s_k1_KO_s011  = 0
 w_k1_KO_s012_return_constraint: 1.17839 w_k1_KO_s01 - w_k1_KO_s012
                                 + 0.99 b_k1_KO_s012 - 1.01 s_k1_KO_s012  = 0
 w_k1_KO_s013_return_constraint: 0.913878 w_k1_KO_s01 - w_k1_KO_s013
                                 + 0.99 b_k1_KO_s013 - 1.01 s_k1_KO_s013  = 0
 w_k1_KO_s014_return_constraint: 0.70694 w_k1_KO_s01 - w_k1_KO_s014
                                 + 0.99 b_k1_KO_s014 - 1.01 s_k1_KO_s014  = 0
 w_k1_KO_s015_return_constraint: 1.10273 w_k1_KO_s01 - w_k1_KO_s015
                                 + 0.99 b_k1_KO_s015 - 1.01 s_k1_KO_s015  = 0
 w_k1_KO_s016_return_constraint: 0.868584 w_k1_KO_s01 - w_k1_KO_s016
                                 + 0.99 b_k1_KO_s016 - 1.01 s_k1_KO_s016  = 0
 w_k1_KO_s017_return_constraint: 1.23749 w_k1_KO_s01 - w_k1_KO_s017
                                 + 0.99 b_k1_KO_s017 - 1.01 s_k1_KO_s017  = 0
 w_k1_KO_s100_return_constraint: 0.695809 w_k1_KO_s10 - w_k1_KO_s100
                                 + 0.99 b_k1_KO_s100 - 1.01 s_k1_KO_s100  = 0
 w_k1_KO_s101_return_constraint: 1.06206 w_k1_KO_s10 - w_k1_KO_s101
                                 + 0.99 b_k1_KO_s101 - 1.01 s_k1_KO_s101  = 0
 w_k1_KO_s102_return_constraint: 1.26312 w_k1_KO_s10 - w_k1_KO_s102
                                 + 0.99 b_k1_KO_s102 - 1.01 s_k1_KO_s102  = 0
 w_k1_KO_s103_return_constraint: 0.956902 w_k1_KO_s10 - w_k1_KO_s103
                                 + 0.99 b_k1_KO_s103 - 1.01 s_k1_KO_s103  = 0
 w_k1_KO_s104_return_constraint: 1.25156 w_k1_KO_s10 - w_k1_KO_s104
                                 + 0.99 b_k1_KO_s104 - 1.01 s_k1_KO_s104  = 0
 w_k1_KO_s105_return_constraint: 1.34339 w_k1_KO_s10 - w_k1_KO_s105
                                 + 0.99 b_k1_KO_s105 - 1.01 s_k1_KO_s105  = 0
 w_k1_KO_s106_return_constraint: 0.817821 w_k1_KO_s10 - w_k1_KO_s106
                                 + 0.99 b_k1_KO_s106 - 1.01 s_k1_KO_s106  = 0
 w_k1_KO_s107_return_constraint: 0.86604 w_k1_KO_s10 - w_k1_KO_s107
                                 + 0.99 b_k1_KO_s107 - 1.01 s_k1_KO_s107  = 0
 w_k1_KO_s110_return_constraint: 0.96214 w_k1_KO_s11 - w_k1_KO_s110
                                 + 0.99 b_k1_KO_s110 - 1.01 s_k1_KO_s110  = 0
 w_k1_KO_s111_return_constraint: 0.734175 w_k1_KO_s11 - w_k1_KO_s111
                                 + 0.99 b_k1_KO_s111 - 1.01 s_k1_KO_s111  = 0
 w_k1_KO_s112_return_constraint: 1.05127 w_k1_KO_s11 - w_k1_KO_s112
                                 + 0.99 b_k1_KO_s112 - 1.01 s_k1_KO_s112  = 0
 w_k1_KO_s113_return_constraint: 1.52082 w_k1_KO_s11 - w_k1_KO_s113
                                 + 0.99 b_k1_KO_s113 - 1.01 s_k1_KO_s113  = 0
 w_k1_KO_s114_return_constraint: 1.28537 w_k1_KO_s11 - w_k1_KO_s114
                                 + 0.99 b_k1_KO_s114 - 1.01 s_k1_KO_s114  = 0
 w_k1_KO_s115_return_constraint: 0.755938 w_k1_KO_s11 - w_k1_KO_s115
                                 + 0.99 b_k1_KO_s115 - 1.01 s_k1_KO_s115  = 0
 w_k1_KO_s116_return_constraint: 0.600681 w_k1_KO_s11 - w_k1_KO_s116
                                 + 0.99 b_k1_KO_s116 - 1.01 s_k1_KO_s116  = 0
 w_k1_KO_s117_return_constraint: 1.0317 w_k1_KO_s11 - w_k1_KO_s117
                                 + 0.99 b_k1_KO_s117 - 1.01 s_k1_KO_s117  = 0
 w_k1_MAT_s0_return_constraint:  1.15177 w_0_MAT - w_k1_MAT_s0
                                 + 0.99 b_k1_MAT_s0 - 1.01 s_k1_MAT_s0  = 0
 w_k1_MAT_s1_return_constraint:  0.989434 w_0_MAT - w_k1_MAT_s1
                                 + 0.99 b_k1_MAT_s1 - 1.01 s_k1_MAT_s1  = 0
 w_k1_MAT_s00_return_constraint: 1.46672 w_k1_MAT_s0 - w_k1_MAT_s00
                                 + 0.99 b_k1_MAT_s00 - 1.01 s_k1_MAT_s00  = 0
 w_k1_MAT_s01_return_constraint: 0.967758 w_k1_MAT_s0 - w_k1_MAT_s01
                                 + 0.99 b_k1_MAT_s01 - 1.01 s_k1_MAT_s01  = 0
 w_k1_MAT_s10_return_constraint: 1.18393 w_k1_MAT_s1 - w_k1_MAT_s10
                                 + 0.99 b_k1_MAT_s10 - 1.01 s_k1_MAT_s10  = 0
 w_k1_MAT_s11_return_constraint: 0.982168 w_k1_MAT_s1 - w_k1_MAT_s11
                                 + 0.99 b_k1_MAT_s11 - 1.01 s_k1_MAT_s11  = 0
 w_k1_MAT_s000_return_constraint: 0.841444 w_k1_MAT_s00 - w_k1_MAT_s000
                                 + 0.99 b_k1_MAT_s000 - 1.01 s_k1_MAT_s000  = 0
 w_k1_MAT_s001_return_constraint: 1.25899 w_k1_MAT_s00 - w_k1_MAT_s001
                                 + 0.99 b_k1_MAT_s001 - 1.01 s_k1_MAT_s001  = 0
 w_k1_MAT_s002_return_constraint: 0.830539 w_k1_MAT_s00 - w_k1_MAT_s002
                                 + 0.99 b_k1_MAT_s002 - 1.01 s_k1_MAT_s002  = 0
 w_k1_MAT_s003_return_constraint: 0.772282 w_k1_MAT_s00 - w_k1_MAT_s003
                                 + 0.99 b_k1_MAT_s003 - 1.01 s_k1_MAT_s003  = 0
 w_k1_MAT_s004_return_constraint: 0.899141 w_k1_MAT_s00 - w_k1_MAT_s004
                                 + 0.99 b_k1_MAT_s004 - 1.01 s_k1_MAT_s004  = 0
 w_k1_MAT_s005_return_constraint: 0.918373 w_k1_MAT_s00 - w_k1_MAT_s005
                                 + 0.99 b_k1_MAT_s005 - 1.01 s_k1_MAT_s005  = 0
 w_k1_MAT_s006_return_constraint: 1.65204 w_k1_MAT_s00 - w_k1_MAT_s006
                                 + 0.99 b_k1_MAT_s006 - 1.01 s_k1_MAT_s006  = 0
 w_k1_MAT_s007_return_constraint: 0.547433 w_k1_MAT_s00 - w_k1_MAT_s007
                                 + 0.99 b_k1_MAT_s007 - 1.01 s_k1_MAT_s007  = 0
 w_k1_MAT_s010_return_constraint: 1.49422 w_k1_MAT_s01 - w_k1_MAT_s010
                                 + 0.99 b_k1_MAT_s010 - 1.01 s_k1_MAT_s010  = 0
 w_k1_MAT_s011_return_constraint: 0.718736 w_k1_MAT_s01 - w_k1_MAT_s011
                                 + 0.99 b_k1_MAT_s011 - 1.01 s_k1_MAT_s011  = 0
 w_k1_MAT_s012_return_constraint: 1.67526 w_k1_MAT_s01 - w_k1_MAT_s012
                                 + 0.99 b_k1_MAT_s012 - 1.01 s_k1_MAT_s012  = 0
 w_k1_MAT_s013_return_constraint: 1.02637 w_k1_MAT_s01 - w_k1_MAT_s013
                                 + 0.99 b_k1_MAT_s013 - 1.01 s_k1_MAT_s013  = 0
 w_k1_MAT_s014_return_constraint: 0.623229 w_k1_MAT_s01 - w_k1_MAT_s014
                                 + 0.99 b_k1_MAT_s014 - 1.01 s_k1_MAT_s014  = 0
 w_k1_MAT_s015_return_constraint: 0.952034 w_k1_MAT_s01 - w_k1_MAT_s015
                                 + 0.99 b_k1_MAT_s015 - 1.01 s_k1_MAT_s015  = 0
 w_k1_MAT_s016_return_constraint: 0.610388 w_k1_MAT_s01 - w_k1_MAT_s016
                                 + 0.99 b_k1_MAT_s016 - 1.01 s_k1_MAT_s016  = 0
 w_k1_MAT_s017_return_constraint: 0.974318 w_k1_MAT_s01 - w_k1_MAT_s017
                                 + 0.99 b_k1_MAT_s017 - 1.01 s_k1_MAT_s017  = 0
 w_k1_MAT_s100_return_constraint: 0.547637 w_k1_MAT_s10 - w_k1_MAT_s100
                                 + 0.99 b_k1_MAT_s100 - 1.01 s_k1_MAT_s100  = 0
 w_k1_MAT_s101_return_constraint: 1.48812 w_k1_MAT_s10 - w_k1_MAT_s101
                                 + 0.99 b_k1_MAT_s101 - 1.01 s_k1_MAT_s101  = 0
 w_k1_MAT_s102_return_constraint: 1.48298 w_k1_MAT_s10 - w_k1_MAT_s102
                                 + 0.99 b_k1_MAT_s102 - 1.01 s_k1_MAT_s102  = 0
 w_k1_MAT_s103_return_constraint: 0.910769 w_k1_MAT_s10 - w_k1_MAT_s103
                                 + 0.99 b_k1_MAT_s103 - 1.01 s_k1_MAT_s103  = 0
 w_k1_MAT_s104_return_constraint: 1.56405 w_k1_MAT_s10 - w_k1_MAT_s104
                                 + 0.99 b_k1_MAT_s104 - 1.01 s_k1_MAT_s104  = 0
 w_k1_MAT_s105_return_constraint: 1.33368 w_k1_MAT_s10 - w_k1_MAT_s105
                                 + 0.99 b_k1_MAT_s105 - 1.01 s_k1_MAT_s105  = 0
 w_k1_MAT_s106_return_constraint: 1.17639 w_k1_MAT_s10 - w_k1_MAT_s106
                                 + 0.99 b_k1_MAT_s106 - 1.01 s_k1_MAT_s106  = 0
 w_k1_MAT_s107_return_constraint: 0.837895 w_k1_MAT_s10 - w_k1_MAT_s107
                                 + 0.99 b_k1_MAT_s107 - 1.01 s_k1_MAT_s107  = 0
 w_k1_MAT_s110_return_constraint: 0.528061 w_k1_MAT_s11 - w_k1_MAT_s110
                                 + 0.99 b_k1_MAT_s110 - 1.01 s_k1_MAT_s110  = 0
 w_k1_MAT_s111_return_constraint: 0.717766 w_k1_MAT_s11 - w_k1_MAT_s111
                                 + 0.99 b_k1_MAT_s111 - 1.01 s_k1_MAT_s111  = 0
 w_k1_MAT_s112_return_constraint: 0.92573 w_k1_MAT_s11 - w_k1_MAT_s112
                                 + 0.99 b_k1_MAT_s112 - 1.01 s_k1_MAT_s112  = 0
 w_k1_MAT_s113_return_constraint: 1.884 w_k1_MAT_s11 - w_k1_MAT_s113
                                 + 0.99 b_k1_MAT_s113 - 1.01 s_k1_MAT_s113  = 0
 w_k1_MAT_s114_return_constraint: 1.22565 w_k1_MAT_s11 - w_k1_MAT_s114
                                 + 0.99 b_k1_MAT_s114 - 1.01 s_k1_MAT_s114  = 0
 w_k1_MAT_s115_return_constraint: 0.533991 w_k1_MAT_s11 - w_k1_MAT_s115
                                 + 0.99 b_k1_MAT_s115 - 1.01 s_k1_MAT_s115  = 0
 w_k1_MAT_s116_return_constraint: 0.435011 w_k1_MAT_s11 - w_k1_MAT_s116
                                 + 0.99 b_k1_MAT_s116 - 1.01 s_k1_MAT_s116  = 0
 w_k1_MAT_s117_return_constraint: 0.860257 w_k1_MAT_s11 - w_k1_MAT_s117
                                 + 0.99 b_k1_MAT_s117 - 1.01 s_k1_MAT_s117  = 0
 w_k1_MSFT_s0_return_constraint: 1.08338 w_0_MSFT - w_k1_MSFT_s0
                                 + 0.99 b_k1_MSFT_s0 - 1.01 s_k1_MSFT_s0  = 0
 w_k1_MSFT_s1_return_constraint: 0.887295 w_0_MSFT - w_k1_MSFT_s1
                                 + 0.99 b_k1_MSFT_s1 - 1.01 s_k1_MSFT_s1  = 0
 w_k1_MSFT_s00_return_constraint: 1.2655 w_k1_MSFT_s0 - w_k1_MSFT_s00
                                 + 0.99 b_k1_MSFT_s00 - 1.01 s_k1_MSFT_s00  = 0
 w_k1_MSFT_s01_return_constraint: 0.989514 w_k1_MSFT_s0 - w_k1_MSFT_s01
                                 + 0.99 b_k1_MSFT_s01 - 1.01 s_k1_MSFT_s01  = 0
 w_k1_MSFT_s10_return_constraint: 1.30173 w_k1_MSFT_s1 - w_k1_MSFT_s10
                                 + 0.99 b_k1_MSFT_s10 - 1.01 s_k1_MSFT_s10  = 0
 w_k1_MSFT_s11_return_constraint: 0.931318 w_k1_MSFT_s1 - w_k1_MSFT_s11
                                 + 0.99 b_k1_MSFT_s11 - 1.01 s_k1_MSFT_s11  = 0
 w_k1_MSFT_s000_return_constraint: 0.740229 w_k1_MSFT_s00 - w_k1_MSFT_s000
                                 + 0.99 b_k1_MSFT_s000 - 1.01 s_k1_MSFT_s000
                                  = 0
 w_k1_MSFT_s001_return_constraint: 1.05078 w_k1_MSFT_s00 - w_k1_MSFT_s001
                                 + 0.99 b_k1_MSFT_s001 - 1.01 s_k1_MSFT_s001
                                  = 0
 w_k1_MSFT_s002_return_constraint: 0.677534 w_k1_MSFT_s00 - w_k1_MSFT_s002
                                 + 0.99 b_k1_MSFT_s002 - 1.01 s_k1_MSFT_s002
                                  = 0
 w_k1_MSFT_s003_return_constraint: 1.06489 w_k1_MSFT_s00 - w_k1_MSFT_s003
                                 + 0.99 b_k1_MSFT_s003 - 1.01 s_k1_MSFT_s003
                                  = 0
 w_k1_MSFT_s004_return_constraint: 1.196 w_k1_MSFT_s00 - w_k1_MSFT_s004
                                 + 0.99 b_k1_MSFT_s004 - 1.01 s_k1_MSFT_s004
                                  = 0
 w_k1_MSFT_s005_return_constraint: 0.833588 w_k1_MSFT_s00 - w_k1_MSFT_s005
                                 + 0.99 b_k1_MSFT_s005 - 1.01 s_k1_MSFT_s005
                                  = 0
 w_k1_MSFT_s006_return_constraint: 0.921778 w_k1_MSFT_s00 - w_k1_MSFT_s006
                                 + 0.99 b_k1_MSFT_s006 - 1.01 s_k1_MSFT_s006
                                  = 0
 w_k1_MSFT_s007_return_constraint: 0.951928 w_k1_MSFT_s00 - w_k1_MSFT_s007
                                 + 0.99 b_k1_MSFT_s007 - 1.01 s_k1_MSFT_s007
                                  = 0
 w_k1_MSFT_s010_return_constraint: 1.19594 w_k1_MSFT_s01 - w_k1_MSFT_s010
                                 + 0.99 b_k1_MSFT_s010 - 1.01 s_k1_MSFT_s010
                                  = 0
 w_k1_MSFT_s011_return_constraint: 0.814578 w_k1_MSFT_s01 - w_k1_MSFT_s011
                                 + 0.99 b_k1_MSFT_s011 - 1.01 s_k1_MSFT_s011
                                  = 0
 w_k1_MSFT_s012_return_constraint: 0.890038 w_k1_MSFT_s01 - w_k1_MSFT_s012
                                 + 0.99 b_k1_MSFT_s012 - 1.01 s_k1_MSFT_s012
                                  = 0
 w_k1_MSFT_s013_return_constraint: 0.850056 w_k1_MSFT_s01 - w_k1_MSFT_s013
                                 + 0.99 b_k1_MSFT_s013 - 1.01 s_k1_MSFT_s013
                                  = 0
 w_k1_MSFT_s014_return_constraint: 0.842819 w_k1_MSFT_s01 - w_k1_MSFT_s014
                                 + 0.99 b_k1_MSFT_s014 - 1.01 s_k1_MSFT_s014
                                  = 0
 w_k1_MSFT_s015_return_constraint: 1.04106 w_k1_MSFT_s01 - w_k1_MSFT_s015
                                 + 0.99 b_k1_MSFT_s015 - 1.01 s_k1_MSFT_s015
                                  = 0
 w_k1_MSFT_s016_return_constraint: 1.08436 w_k1_MSFT_s01 - w_k1_MSFT_s016
                                 + 0.99 b_k1_MSFT_s016 - 1.01 s_k1_MSFT_s016
                                  = 0
 w_k1_MSFT_s017_return_constraint: 1.38744 w_k1_MSFT_s01 - w_k1_MSFT_s017
                                 + 0.99 b_k1_MSFT_s017 - 1.01 s_k1_MSFT_s017
                                  = 0
 w_k1_MSFT_s100_return_constraint: 0.947229 w_k1_MSFT_s10 - w_k1_MSFT_s100
                                 + 0.99 b_k1_MSFT_s100 - 1.01 s_k1_MSFT_s100
                                  = 0
 w_k1_MSFT_s101_return_constraint: 1.07409 w_k1_MSFT_s10 - w_k1_MSFT_s101
                                 + 0.99 b_k1_MSFT_s101 - 1.01 s_k1_MSFT_s101
                                  = 0
 w_k1_MSFT_s102_return_constraint: 0.831147 w_k1_MSFT_s10 - w_k1_MSFT_s102
                                 + 0.99 b_k1_MSFT_s102 - 1.01 s_k1_MSFT_s102
                                  = 0
 w_k1_MSFT_s103_return_constraint: 0.907152 w_k1_MSFT_s10 - w_k1_MSFT_s103
                                 + 0.99 b_k1_MSFT_s103 - 1.01 s_k1_MSFT_s103
                                  = 0
 w_k1_MSFT_s104_return_constraint: 1.36601 w_k1_MSFT_s10 - w_k1_MSFT_s104
                                 + 0.99 b_k1_MSFT_s104 - 1.01 s_k1_MSFT_s104
                                  = 0
 w_k1_MSFT_s105_return_constraint: 1.19374 w_k1_MSFT_s10 - w_k1_MSFT_s105
                                 + 0.99 b_k1_MSFT_s105 - 1.01 s_k1_MSFT_s105
                                  = 0
 w_k1_MSFT_s106_return_constraint: 0.821568 w_k1_MSFT_s10 - w_k1_MSFT_s106
                                 + 0.99 b_k1_MSFT_s106 - 1.01 s_k1_MSFT_s106
                                  = 0
 w_k1_MSFT_s107_return_constraint: 0.829002 w_k1_MSFT_s10 - w_k1_MSFT_s107
                                 + 0.99 b_k1_MSFT_s107 - 1.01 s_k1_MSFT_s107
                                  = 0
 w_k1_MSFT_s110_return_constraint: 1.19134 w_k1_MSFT_s11 - w_k1_MSFT_s110
                                 + 0.99 b_k1_MSFT_s110 - 1.01 s_k1_MSFT_s110
                                  = 0
 w_k1_MSFT_s111_return_constraint: 0.686603 w_k1_MSFT_s11 - w_k1_MSFT_s111
                                 + 0.99 b_k1_MSFT_s111 - 1.01 s_k1_MSFT_s111
                                  = 0
 w_k1_MSFT_s112_return_constraint: 0.933706 w_k1_MSFT_s11 - w_k1_MSFT_s112
                                 + 0.99 b_k1_MSFT_s112 - 1.01 s_k1_MSFT_s112
                                  = 0
 w_k1_MSFT_s113_return_constraint: 0.965462 w_k1_MSFT_s11 - w_k1_MSFT_s113
                                 + 0.99 b_k1_MSFT_s113 - 1.01 s_k1_MSFT_s113
                                  = 0
 w_k1_MSFT_s114_return_constraint: 1.29096 w_k1_MSFT_s11 - w_k1_MSFT_s114
                                 + 0.99 b_k1_MSFT_s114 - 1.01 s_k1_MSFT_s114
                                  = 0
 w_k1_MSFT_s115_return_constraint: 0.844897 w_k1_MSFT_s11 - w_k1_MSFT_s115
                                 + 0.99 b_k1_MSFT_s115 - 1.01 s_k1_MSFT_s115
                                  = 0
 w_k1_MSFT_s116_return_constraint: 0.744184 w_k1_MSFT_s11 - w_k1_MSFT_s116
                                 + 0.99 b_k1_MSFT_s116 - 1.01 s_k1_MSFT_s116
                                  = 0
 w_k1_MSFT_s117_return_constraint: 0.965296 w_k1_MSFT_s11 - w_k1_MSFT_s117
                                 + 0.99 b_k1_MSFT_s117 - 1.01 s_k1_MSFT_s117
                                  = 0
 w_k1_PG_s0_return_constraint:   1.10468 w_0_PG - w_k1_PG_s0 + 0.99 b_k1_PG_s0
                                 - 1.01 s_k1_PG_s0  = 0
 w_k1_PG_s1_return_constraint:   0.896102 w_0_PG - w_k1_PG_s1 + 0.99 b_k1_PG_s1
                                 - 1.01 s_k1_PG_s1  = 0
 w_k1_PG_s00_return_constraint:  1.14839 w_k1_PG_s0 - w_k1_PG_s00
                                 + 0.99 b_k1_PG_s00 - 1.01 s_k1_PG_s00  = 0
 w_k1_PG_s01_return_constraint:  0.87617 w_k1_PG_s0 - w_k1_PG_s01
                                 + 0.99 b_k1_PG_s01 - 1.01 s_k1_PG_s01  = 0
 w_k1_PG_s10_return_constraint:  1.26877 w_k1_PG_s1 - w_k1_PG_s10
                                 + 0.99 b_k1_PG_s10 - 1.01 s_k1_PG_s10  = 0
 w_k1_PG_s11_return_constraint:  1.03809 w_k1_PG_s1 - w_k1_PG_s11
                                 + 0.99 b_k1_PG_s11 - 1.01 s_k1_PG_s11  = 0
 w_k1_PG_s000_return_constraint: 0.857808 w_k1_PG_s00 - w_k1_PG_s000
                                 + 0.99 b_k1_PG_s000 - 1.01 s_k1_PG_s000  = 0
 w_k1_PG_s001_return_constraint: 1.01887 w_k1_PG_s00 - w_k1_PG_s001
                                 + 0.99 b_k1_PG_s001 - 1.01 s_k1_PG_s001  = 0
 w_k1_PG_s002_return_constraint: 0.796592 w_k1_PG_s00 - w_k1_PG_s002
                                 + 0.99 b_k1_PG_s002 - 1.01 s_k1_PG_s002  = 0
 w_k1_PG_s003_return_constraint: 0.926563 w_k1_PG_s00 - w_k1_PG_s003
                                 + 0.99 b_k1_PG_s003 - 1.01 s_k1_PG_s003  = 0
 w_k1_PG_s004_return_constraint: 1.16045 w_k1_PG_s00 - w_k1_PG_s004
                                 + 0.99 b_k1_PG_s004 - 1.01 s_k1_PG_s004  = 0
 w_k1_PG_s005_return_constraint: 0.938182 w_k1_PG_s00 - w_k1_PG_s005
                                 + 0.99 b_k1_PG_s005 - 1.01 s_k1_PG_s005  = 0
 w_k1_PG_s006_return_constraint: 1.17682 w_k1_PG_s00 - w_k1_PG_s006
                                 + 0.99 b_k1_PG_s006 - 1.01 s_k1_PG_s006  = 0
 w_k1_PG_s007_return_constraint: 0.85108 w_k1_PG_s00 - w_k1_PG_s007
                                 + 0.99 b_k1_PG_s007 - 1.01 s_k1_PG_s007  = 0
 w_k1_PG_s010_return_constraint: 1.22128 w_k1_PG_s01 - w_k1_PG_s010
                                 + 0.99 b_k1_PG_s010 - 1.01 s_k1_PG_s010  = 0
 w_k1_PG_s011_return_constraint: 0.932155 w_k1_PG_s01 - w_k1_PG_s011
                                 + 0.99 b_k1_PG_s011 - 1.01 s_k1_PG_s011  = 0
 w_k1_PG_s012_return_constraint: 1.12953 w_k1_PG_s01 - w_k1_PG_s012
                                 + 0.99 b_k1_PG_s012 - 1.01 s_k1_PG_s012  = 0
 w_k1_PG_s013_return_constraint: 0.950916 w_k1_PG_s01 - w_k1_PG_s013
                                 + 0.99 b_k1_PG_s013 - 1.01 s_k1_PG_s013  = 0
 w_k1_PG_s014_return_constraint: 0.92194 w_k1_PG_s01 - w_k1_PG_s014
                                 + 0.99 b_k1_PG_s014 - 1.01 s_k1_PG_s014  = 0
 w_k1_PG_s015_return_constraint: 1.11195 w_k1_PG_s01 - w_k1_PG_s015
                                 + 0.99 b_k1_PG_s015 - 1.01 s_k1_PG_s015  = 0
 w_k1_PG_s016_return_constraint: 1.07076 w_k1_PG_s01 - w_k1_PG_s016
                                 + 0.99 b_k1_PG_s016 - 1.01 s_k1_PG_s016  = 0
 w_k1_PG_s017_return_constraint: 1.36713 w_k1_PG_s01 - w_k1_PG_s017
                                 + 0.99 b_k1_PG_s017 - 1.01 s_k1_PG_s017  = 0
 w_k1_PG_s100_return_constraint: 0.845226 w_k1_PG_s10 - w_k1_PG_s100
                                 + 0.99 b_k1_PG_s100 - 1.01 s_k1_PG_s100  = 0
 w_k1_PG_s101_return_constraint: 1.00474 w_k1_PG_s10 - w_k1_PG_s101
                                 + 0.99 b_k1_PG_s101 - 1.01 s_k1_PG_s101  = 0
 w_k1_PG_s102_return_constraint: 0.972669 w_k1_PG_s10 - w_k1_PG_s102
                                 + 0.99 b_k1_PG_s102 - 1.01 s_k1_PG_s102  = 0
 w_k1_PG_s103_return_constraint: 0.976648 w_k1_PG_s10 - w_k1_PG_s103
                                 + 0.99 b_k1_PG_s103 - 1.01 s_k1_PG_s103  = 0
 w_k1_PG_s104_return_constraint: 1.18056 w_k1_PG_s10 - w_k1_PG_s104
                                 + 0.99 b_k1_PG_s104 - 1.01 s_k1_PG_s104  = 0
 w_k1_PG_s105_return_constraint: 1.11193 w_k1_PG_s10 - w_k1_PG_s105
                                 + 0.99 b_k1_PG_s105 - 1.01 s_k1_PG_s105  = 0
 w_k1_PG_s106_return_constraint: 0.873515 w_k1_PG_s10 - w_k1_PG_s106
                                 + 0.99 b_k1_PG_s106 - 1.01 s_k1_PG_s106  = 0
 w_k1_PG_s107_return_constraint: 0.865604 w_k1_PG_s10 - w_k1_PG_s107
                                 + 0.99 b_k1_PG_s107 - 1.01 s_k1_PG_s107  = 0
 w_k1_PG_s110_return_constraint: 1.02078 w_k1_PG_s11 - w_k1_PG_s110
                                 + 0.99 b_k1_PG_s110 - 1.01 s_k1_PG_s110  = 0
 w_k1_PG_s111_return_constraint: 0.756759 w_k1_PG_s11 - w_k1_PG_s111
                                 + 0.99 b_k1_PG_s111 - 1.01 s_k1_PG_s111  = 0
 w_k1_PG_s112_return_constraint: 0.920411 w_k1_PG_s11 - w_k1_PG_s112
                                 + 0.99 b_k1_PG_s112 - 1.01 s_k1_PG_s112  = 0
 w_k1_PG_s113_return_constraint: 1.06118 w_k1_PG_s11 - w_k1_PG_s113
                                 + 0.99 b_k1_PG_s113 - 1.01 s_k1_PG_s113  = 0
 w_k1_PG_s114_return_constraint: 1.17258 w_k1_PG_s11 - w_k1_PG_s114
                                 + 0.99 b_k1_PG_s114 - 1.01 s_k1_PG_s114  = 0
 w_k1_PG_s115_return_constraint: 0.815032 w_k1_PG_s11 - w_k1_PG_s115
                                 + 0.99 b_k1_PG_s115 - 1.01 s_k1_PG_s115  = 0
 w_k1_PG_s116_return_constraint: 0.69185 w_k1_PG_s11 - w_k1_PG_s116
                                 + 0.99 b_k1_PG_s116 - 1.01 s_k1_PG_s116  = 0
 w_k1_PG_s117_return_constraint: 1.02275 w_k1_PG_s11 - w_k1_PG_s117
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
 k1_wcvar_constraint:            wcvar - var_1 - 0.153000242437847 z_k1_s0000
                                 - 0.258000295973539 z_k1_s0001
                                 - 0.0810000741847018 z_k1_s0002
                                 - 0.0700000740571572 z_k1_s0003
                                 - 0.116000168763323 z_k1_s0004
                                 - 0.144000566706462 z_k1_s0005
                                 - 0.100000105795939 z_k1_s0006
                                 - 0.228999678282278 z_k1_s0007
                                 - 0.228000210612001 z_k1_s0010
                                 - 0.686000633683477 z_k1_s0011
                                 - 0.416000384274528 z_k1_s0012
                                 - 0.700000646615793 z_k1_s0013
                                 - 0.579000534843634 z_k1_s0014
                                 - 0.478000441546213 z_k1_s0015
                                 - 0.671000619827424 z_k1_s0016
                                 - 0.242000223544317 z_k1_s0017
                                 - 0.0770003993415554 z_k1_s0020
                                 - 0.362001636238425 z_k1_s0021
                                 - 0.278001614704063 z_k1_s0022
                                 - 0.158000632850656 z_k1_s0023
                                 - 0.206001095672337 z_k1_s0024
                                 - 0.35800180207088 z_k1_s0025
                                 - 0.171000707097962 z_k1_s0026
                                 - 0.142000595377293 z_k1_s0027
                                 - 0.635001998767351 z_k1_s0030
                                 - 0.384000950277949 z_k1_s0031
                                 - 0.213000373263395 z_k1_s0032
                                 - 0.427001757533737 z_k1_s0033
                                 - 0.330999878962599 z_k1_s0034
                                 - 0.677999524019837 z_k1_s0035
                                 - 0.303000452397145 z_k1_s0036
                                 - 0.311000335444465 z_k1_s0037
                                 - 0.272000222999045 z_k1_s0040
                                 - 0.107000360156512 z_k1_s0041
                                 - 0.429001608517839 z_k1_s0042
                                 - 0.161000624180663 z_k1_s0043
                                 - 0.354001298678951 z_k1_s0044
                                 - 0.109000491239336 z_k1_s0045
                                 - 0.399000665779261 z_k1_s0046
                                 - 0.216000237293582 z_k1_s0047
                                 - 0.100999785276838 z_k1_s0050
                                 - 0.151999647918673 z_k1_s0051
                                 - 0.382999197666595 z_k1_s0052
                                 - 0.159999683672163 z_k1_s0053
                                 - 0.364999589945158 z_k1_s0054
                                 - 0.273999376636267 z_k1_s0055
                                 - 0.123999780630867 z_k1_s0056
                                 - 0.159999683672163 z_k1_s0057
                                 - 0.197999888945618 z_k1_s0060
                                 - 0.12900011551321 z_k1_s0061
                                 - 0.142999528404845 z_k1_s0062
                                 - 0.337999653866633 z_k1_s0063
                                 - 0.0709999914886062 z_k1_s0064
                                 - 0.193999734633456 z_k1_s0065
                                 - 0.197999888945618 z_k1_s0066
                                 - 0.138000110465283 z_k1_s0067
                                 - 0.146000084865534 z_k1_s0070
                                 - 0.292999770654434 z_k1_s0071
                                 - 0.0850001285176782 z_k1_s0072
                                 - 0.238999720772645 z_k1_s0073
                                 - 0.0800000238989016 z_k1_s0074
                                 - 0.201000185671106 z_k1_s0075
                                 - 0.28300076141799 z_k1_s0076
                                 - 0.172999659806013 z_k1_s0077
                                 - 0.45899994563458 z_k1_s0100
                                 - 0.54499825043469 z_k1_s0101
                                 - 0.315000163552968 z_k1_s0102
                                 - 0.290998492540859 z_k1_s0103
                                 - 0.19499974758548 z_k1_s0104
                                 - 0.238999737910367 z_k1_s0105
                                 - 0.31099988505095 z_k1_s0106
                                 - 0.205999745166702 z_k1_s0107
                                 - 1.01899800183083 z_k1_s0110
                                 - 0.938999602149951 z_k1_s0111
                                 - 0.489998403321117 z_k1_s0112
                                 - 1.0369949066671 z_k1_s0113
                                 - 0.617997058406926 z_k1_s0114
                                 - 0.760999819969881 z_k1_s0115
                                 - 0.425998467980009 z_k1_s0116
                                 - 0.787998216215318 z_k1_s0117
                                 - 0.245000088517349 z_k1_s0120
                                 - 0.841002944184846 z_k1_s0121
                                 - 0.562002278687425 z_k1_s0122
                                 - 0.5089990697693 z_k1_s0123
                                 - 0.418000412384126 z_k1_s0124
                                 - 0.258000364115662 z_k1_s0125
                                 - 0.770000144654401 z_k1_s0126
                                 - 1.07100134845673 z_k1_s0127
                                 - 0.554999081830569 z_k1_s0130
                                 - 0.17999989880453 z_k1_s0131
                                 - 0.655999846718755 z_k1_s0132
                                 - 0.893000481268136 z_k1_s0133
                                 - 0.491001757931466 z_k1_s0134
                                 - 0.433001278678839 z_k1_s0135
                                 - 0.537000465890265 z_k1_s0136
                                 - 0.296000049109695 z_k1_s0137
                                 - 0.673000094054657 z_k1_s0140
                                 - 0.153999976785204 z_k1_s0141
                                 - 0.233999761699763 z_k1_s0142
                                 - 0.290999369614084 z_k1_s0143
                                 - 0.394000489527925 z_k1_s0144
                                 - 0.640998790489981 z_k1_s0145
                                 - 0.137999904002387 z_k1_s0146
                                 - 0.370000959353222 z_k1_s0147
                                 - 0.196999694302579 z_k1_s0150
                                 - 0.624998408285316 z_k1_s0151
                                 - 0.368001196009536 z_k1_s0152
                                 - 0.28299985966794 z_k1_s0153
                                 - 0.44400002494739 z_k1_s0154
                                 - 0.191999821588208 z_k1_s0155
                                 - 0.388999197691034 z_k1_s0156
                                 - 0.683999006433268 z_k1_s0157
                                 - 0.336999197090343 z_k1_s0160
                                 - 1.55999626605325 z_k1_s0161
                                 - 0.984996396622441 z_k1_s0162
                                 - 1.06299649997457 z_k1_s0163
                                 - 0.357999358784903 z_k1_s0164
                                 - 0.525998912044566 z_k1_s0165
                                 - 0.608999468251788 z_k1_s0166
                                 - 0.362999231541054 z_k1_s0167
                                 - 0.512999343205182 z_k1_s0170
                                 - 0.540999850317893 z_k1_s0171
                                 - 0.476998167633432 z_k1_s0172
                                 - 0.430998905228767 z_k1_s0173
                                 - 0.284998983560596 z_k1_s0174
                                 - 0.361998912125416 z_k1_s0175
                                 - 0.80499625455125 z_k1_s0176
                                 - 0.25099925784935 z_k1_s0177
                                 - 0.349999538598606 z_k1_s1000
                                 - 0.224999949241799 z_k1_s1001
                                 - 0.210999623498082 z_k1_s1002
                                 - 0.18900009504302 z_k1_s1003
                                 - 0.133999897106275 z_k1_s1004
                                 - 0.279999630878885 z_k1_s1005
                                 - 0.139999987539329 z_k1_s1006
                                 - 0.191999796059774 z_k1_s1007
                                 - 0.483000029817835 z_k1_s1010
                                 - 0.341000021051515 z_k1_s1011
                                 - 0.17000001049489 z_k1_s1012
                                 - 0.14200000876632 z_k1_s1013
                                 - 0.502000030990793 z_k1_s1014
                                 - 0.154000009507136 z_k1_s1015
                                 - 0.106000006543873 z_k1_s1016
                                 - 0.102000006296934 z_k1_s1017
                                 - 0.0439999519587423 z_k1_s1020
                                 - 0.0900000708244725 z_k1_s1021
                                 - 0.105999769282579 z_k1_s1022
                                 - 0.132999492930322 z_k1_s1023
                                 - 0.148999935387427 z_k1_s1024
                                 - 0.104999834628885 z_k1_s1025
                                 - 0.0549999027484779 z_k1_s1026
                                 - 0.0619998917237369 z_k1_s1027
                                 - 0.130000030456044 z_k1_s1030
                                 - 0.0879998534781249 z_k1_s1031
                                 - 0.0879998534781249 z_k1_s1032
                                 - 0.17599970695625 z_k1_s1033
                                 - 0.0539999171433869 z_k1_s1034
                                 - 0.0729999454715513 z_k1_s1035
                                 - 0.102999916684525 z_k1_s1036
                                 - 0.0639999075477114 z_k1_s1037
                                 - 0.0719998207809594 z_k1_s1040
                                 - 0.176999965443912 z_k1_s1041
                                 - 0.0809997872036054 z_k1_s1042
                                 - 0.0539998879356674 z_k1_s1043
                                 - 0.209999216528406 z_k1_s1044
                                 - 0.119999552301946 z_k1_s1045
                                 - 0.119000062631583 z_k1_s1046
                                 - 0.0609998916198782 z_k1_s1047
                                 - 0.110999774152879 z_k1_s1050
                                 - 0.107999751975836 z_k1_s1051
                                 - 0.136999617554683 z_k1_s1052
                                 - 0.370998992969187 z_k1_s1053
                                 - 0.0989998162444202 z_k1_s1054
                                 - 0.090999844305448 z_k1_s1055
                                 - 0.125999754238379 z_k1_s1056
                                 - 0.264998906978816 z_k1_s1057
                                 - 0.0339999276268343 z_k1_s1060
                                 - 0.110999845340433 z_k1_s1061
                                 - 0.0739999218935874 z_k1_s1062
                                 - 0.0879996273873586 z_k1_s1063
                                 - 0.0729998732869865 z_k1_s1064
                                 - 0.122000080013461 z_k1_s1065
                                 - 0.083999882960328 z_k1_s1066
                                 - 0.164000021493625 z_k1_s1067
                                 - 0.477998308968644 z_k1_s1070
                                 - 0.389998313008463 z_k1_s1071
                                 - 0.452998835569886 z_k1_s1072
                                 - 0.407999239453135 z_k1_s1073
                                 - 0.137999758756178 z_k1_s1074
                                 - 0.338000169937625 z_k1_s1075
                                 - 0.127999697397181 z_k1_s1076
                                 - 0.386998158601017 z_k1_s1077
                                 - 0.793999326518741 z_k1_s1100
                                 - 0.985997422472714 z_k1_s1101
                                 - 0.378999912588693 z_k1_s1102
                                 - 0.389999886131898 z_k1_s1103
                                 - 0.66900168752799 z_k1_s1104
                                 - 0.763997853400749 z_k1_s1105
                                 - 0.92500292613368 z_k1_s1106
                                 - 0.759000440881378 z_k1_s1107
                                 - 0.114000227995749 z_k1_s1110
                                 - 0.315001341910146 z_k1_s1111
                                 - 0.533999797473323 z_k1_s1112
                                 - 0.184000156242011 z_k1_s1113
                                 - 0.114000227995749 z_k1_s1114
                                 - 0.3329999321604 z_k1_s1115
                                 - 0.151000243580265 z_k1_s1116
                                 - 0.336000737703337 z_k1_s1117
                                 - 0.897003793742601 z_k1_s1120
                                 - 0.859005293813527 z_k1_s1121
                                 - 0.957005474377891 z_k1_s1122
                                 - 0.524001622047881 z_k1_s1123
                                 - 1.20899994381366 z_k1_s1124
                                 - 0.433001054722799 z_k1_s1125
                                 - 0.708004444656761 z_k1_s1126
                                 - 1.4070005940975 z_k1_s1127
                                 - 0.640001516403163 z_k1_s1130
                                 - 0.612999426599544 z_k1_s1131
                                 - 0.209000232684493 z_k1_s1132
                                 - 0.235000284382111 z_k1_s1133
                                 - 0.323001166513289 z_k1_s1134
                                 - 0.196000365335859 z_k1_s1135
                                 - 0.5189993128467 z_k1_s1136
                                 - 0.435000560132881 z_k1_s1137
                                 - 0.873003711572499 z_k1_s1140
                                 - 1.29400226931405 z_k1_s1141
                                 - 0.618001407210212 z_k1_s1142
                                 - 0.851999754610094 z_k1_s1143
                                 - 0.854005038048392 z_k1_s1144
                                 - 0.528001433131681 z_k1_s1145
                                 - 0.827999761522486 z_k1_s1146
                                 - 2.07900336422482 z_k1_s1147
                                 - 0.533003307869871 z_k1_s1150
                                 - 0.60700498162806 z_k1_s1151
                                 - 0.307001509649449 z_k1_s1152
                                 - 0.689005113298748 z_k1_s1153
                                 - 0.880002011102606 z_k1_s1154
                                 - 0.958005365827441 z_k1_s1155
                                 - 0.615003439540559 z_k1_s1156
                                 - 0.315001438768185 z_k1_s1157
                                 - 0.27899919468893 z_k1_s1160
                                 - 0.473999821837639 z_k1_s1161
                                 - 0.208000141958387 z_k1_s1162
                                 - 0.154999991771824 z_k1_s1163
                                 - 0.243000072100382 z_k1_s1164
                                 - 0.349999301419944 z_k1_s1165
                                 - 0.365999028570463 z_k1_s1166
                                 - 0.560000990272853 z_k1_s1167
                                 - 1.57400322593319 z_k1_s1170
                                 - 1.29900516648329 z_k1_s1171
                                 - 0.70500292773603 z_k1_s1172
                                 - 0.626999386114942 z_k1_s1173
                                 - 0.341000689805326 z_k1_s1174
                                 - 0.387000809048922 z_k1_s1175
                                 - 0.51400112532143 z_k1_s1176
                                 - 0.507000990684081 z_k1_s1177 >= 0
 z_k1_s0000_constraint:          var_1 + z_k1_s0000 + 1.00539 w_k1_ADM_s000
                                 + 1.19608 w_k1_AXP_s000
                                 + 0.967419 w_k1_DIS_s000
                                 + 0.968856 w_k1_FDX_s000
                                 + 1.03316 w_k1_IBM_s000
                                 + 0.921006 w_k1_INTC_s000
                                 + 1.00746 w_k1_KO_s000
                                 + 0.913191 w_k1_MAT_s000
                                 + 1.008 w_k1_MSFT_s000 + 0.977298 w_k1_PG_s000
                                 >= 1
 z_k1_s0001_constraint:          var_1 + z_k1_s0001 + 0.910067 w_k1_ADM_s000
                                 + 1.29543 w_k1_AXP_s000
                                 + 1.01073 w_k1_DIS_s000
                                 + 1.11091 w_k1_FDX_s000
                                 + 0.818999 w_k1_IBM_s000
                                 + 1.18694 w_k1_INTC_s000
                                 + 0.914126 w_k1_KO_s000
                                 + 0.934639 w_k1_MAT_s000
                                 + 1.19483 w_k1_MSFT_s000
                                 + 1.01239 w_k1_PG_s000 >= 1
 z_k1_s0002_constraint:          var_1 + z_k1_s0002 + 0.824183 w_k1_ADM_s000
                                 + 0.832221 w_k1_AXP_s000
                                 + 0.973411 w_k1_DIS_s000
                                 + 0.952538 w_k1_FDX_s000
                                 + 1.23407 w_k1_IBM_s000
                                 + 1.09614 w_k1_INTC_s000
                                 + 1.15245 w_k1_KO_s000 + 1.1583 w_k1_MAT_s000
                                 + 1.11409 w_k1_MSFT_s000
                                 + 1.09188 w_k1_PG_s000 >= 1
 z_k1_s0003_constraint:          var_1 + z_k1_s0003 + 0.860185 w_k1_ADM_s000
                                 + 0.773516 w_k1_AXP_s000
                                 + 0.759988 w_k1_DIS_s000
                                 + 0.768866 w_k1_FDX_s000
                                 + 1.05445 w_k1_IBM_s000
                                 + 0.683379 w_k1_INTC_s000
                                 + 0.794498 w_k1_KO_s000
                                 + 0.996174 w_k1_MAT_s000
                                 + 0.838695 w_k1_MSFT_s000
                                 + 0.902744 w_k1_PG_s000 >= 1
 z_k1_s0004_constraint:          var_1 + z_k1_s0004 + 0.825317 w_k1_ADM_s000
                                 + 0.822766 w_k1_AXP_s000
                                 + 0.725216 w_k1_DIS_s000
                                 + 0.820158 w_k1_FDX_s000
                                 + 0.849219 w_k1_IBM_s000
                                 + 0.999977 w_k1_INTC_s000
                                 + 0.823755 w_k1_KO_s000
                                 + 0.792721 w_k1_MAT_s000
                                 + 0.989665 w_k1_MSFT_s000
                                 + 0.98548 w_k1_PG_s000 >= 1
 z_k1_s0005_constraint:          var_1 + z_k1_s0005 + 0.893683 w_k1_ADM_s000
                                 + 1.25818 w_k1_AXP_s000
                                 + 1.11126 w_k1_DIS_s000
                                 + 1.01134 w_k1_FDX_s000 + 1.1839 w_k1_IBM_s000
                                 + 0.903804 w_k1_INTC_s000
                                 + 1.09632 w_k1_KO_s000 + 1.39362 w_k1_MAT_s000
                                 + 1.05751 w_k1_MSFT_s000 + 1.119 w_k1_PG_s000
                                 >= 1
 z_k1_s0006_constraint:          var_1 + z_k1_s0006 + 0.831134 w_k1_ADM_s000
                                 + 0.726713 w_k1_AXP_s000
                                 + 0.688516 w_k1_DIS_s000
                                 + 0.8275 w_k1_FDX_s000
                                 + 0.665099 w_k1_IBM_s000
                                 + 0.742042 w_k1_INTC_s000
                                 + 0.729583 w_k1_KO_s000
                                 + 0.807908 w_k1_MAT_s000
                                 + 0.750596 w_k1_MSFT_s000
                                 + 0.864836 w_k1_PG_s000 >= 1
 z_k1_s0007_constraint:          var_1 + z_k1_s0007 + 1.08508 w_k1_ADM_s000
                                 + 1.31459 w_k1_AXP_s000
                                 + 1.45902 w_k1_DIS_s000 + 1.1249 w_k1_FDX_s000
                                 + 1.40666 w_k1_IBM_s000
                                 + 1.21559 w_k1_INTC_s000
                                 + 1.27614 w_k1_KO_s000 + 1.36496 w_k1_MAT_s000
                                 + 1.35629 w_k1_MSFT_s000
                                 + 1.17871 w_k1_PG_s000 >= 1
 z_k1_s0010_constraint:          var_1 + z_k1_s0010 + 1.21896 w_k1_ADM_s001
                                 + 1.50383 w_k1_AXP_s001
                                 + 1.83532 w_k1_DIS_s001
                                 + 1.49266 w_k1_FDX_s001
                                 + 1.21742 w_k1_IBM_s001
                                 + 1.01111 w_k1_INTC_s001 + 1.4573 w_k1_KO_s001
                                 + 1.63262 w_k1_MAT_s001
                                 + 1.30003 w_k1_MSFT_s001 + 1.523 w_k1_PG_s001
                                 >= 1
 z_k1_s0011_constraint:          var_1 + z_k1_s0011 + 0.958625 w_k1_ADM_s001
                                 + 0.900933 w_k1_AXP_s001
                                 + 0.833772 w_k1_DIS_s001
                                 + 1.00636 w_k1_FDX_s001
                                 + 0.761313 w_k1_IBM_s001
                                 + 0.848865 w_k1_INTC_s001
                                 + 0.788386 w_k1_KO_s001
                                 + 0.596864 w_k1_MAT_s001
                                 + 0.959505 w_k1_MSFT_s001
                                 + 0.985537 w_k1_PG_s001 >= 1
 z_k1_s0012_constraint:          var_1 + z_k1_s0012 + 0.899014 w_k1_ADM_s001
                                 + 1.02568 w_k1_AXP_s001 + 1.3754 w_k1_DIS_s001
                                 + 1.25666 w_k1_FDX_s001
                                 + 0.80308 w_k1_IBM_s001
                                 + 1.10221 w_k1_INTC_s001
                                 + 1.09138 w_k1_KO_s001
                                 + 0.872212 w_k1_MAT_s001
                                 + 1.33619 w_k1_MSFT_s001
                                 + 1.23831 w_k1_PG_s001 >= 1
 z_k1_s0013_constraint:          var_1 + z_k1_s0013 + 1.29435 w_k1_ADM_s001
                                 + 1.43064 w_k1_AXP_s001
                                 + 1.32364 w_k1_DIS_s001
                                 + 1.13366 w_k1_FDX_s001
                                 + 1.29263 w_k1_IBM_s001
                                 + 0.795898 w_k1_INTC_s001
                                 + 1.22782 w_k1_KO_s001 + 1.42537 w_k1_MAT_s001
                                 + 1.20007 w_k1_MSFT_s001
                                 + 1.24103 w_k1_PG_s001 >= 1
 z_k1_s0014_constraint:          var_1 + z_k1_s0014 + 0.865729 w_k1_ADM_s001
                                 + 0.955725 w_k1_AXP_s001
                                 + 0.876901 w_k1_DIS_s001
                                 + 0.908351 w_k1_FDX_s001
                                 + 1.36243 w_k1_IBM_s001
                                 + 0.7008 w_k1_INTC_s001 + 1.05213 w_k1_KO_s001
                                 + 1.48144 w_k1_MAT_s001
                                 + 0.905035 w_k1_MSFT_s001
                                 + 1.08784 w_k1_PG_s001 >= 1
 z_k1_s0015_constraint:          var_1 + z_k1_s0015 + 0.966334 w_k1_ADM_s001
                                 + 1.06208 w_k1_AXP_s001
                                 + 0.797591 w_k1_DIS_s001
                                 + 0.96564 w_k1_FDX_s001
                                 + 0.98305 w_k1_IBM_s001
                                 + 0.666043 w_k1_INTC_s001
                                 + 0.827165 w_k1_KO_s001
                                 + 0.918241 w_k1_MAT_s001
                                 + 0.787545 w_k1_MSFT_s001
                                 + 0.951874 w_k1_PG_s001 >= 1
 z_k1_s0016_constraint:          var_1 + z_k1_s0016 + 0.833174 w_k1_ADM_s001
                                 + 0.830743 w_k1_AXP_s001
                                 + 0.60893 w_k1_DIS_s001
                                 + 0.708707 w_k1_FDX_s001
                                 + 0.933049 w_k1_IBM_s001
                                 + 0.721735 w_k1_INTC_s001
                                 + 0.76356 w_k1_KO_s001
                                 + 0.881572 w_k1_MAT_s001
                                 + 0.83084 w_k1_MSFT_s001
                                 + 0.848691 w_k1_PG_s001 >= 1
 z_k1_s0017_constraint:          var_1 + z_k1_s0017 + 1.32792 w_k1_ADM_s001
                                 + 1.40787 w_k1_AXP_s001
                                 + 1.14868 w_k1_DIS_s001
                                 + 1.37011 w_k1_FDX_s001
                                 + 0.967489 w_k1_IBM_s001
                                 + 0.89214 w_k1_INTC_s001
                                 + 0.983182 w_k1_KO_s001 + 1.0063 w_k1_MAT_s001
                                 + 0.941927 w_k1_MSFT_s001
                                 + 1.17947 w_k1_PG_s001 >= 1
 z_k1_s0020_constraint:          var_1 + z_k1_s0020 + 1.02782 w_k1_ADM_s002
                                 + 0.760179 w_k1_AXP_s002
                                 + 0.840728 w_k1_DIS_s002
                                 + 0.980837 w_k1_FDX_s002
                                 + 0.647786 w_k1_IBM_s002
                                 + 0.799817 w_k1_INTC_s002
                                 + 0.734788 w_k1_KO_s002
                                 + 0.629818 w_k1_MAT_s002
                                 + 0.823826 w_k1_MSFT_s002
                                 + 0.78588 w_k1_PG_s002 >= 1
 z_k1_s0021_constraint:          var_1 + z_k1_s0021 + 1.05328 w_k1_ADM_s002
                                 + 0.810239 w_k1_AXP_s002
                                 + 0.712081 w_k1_DIS_s002
                                 + 0.855387 w_k1_FDX_s002
                                 + 1.02554 w_k1_IBM_s002
                                 + 0.849074 w_k1_INTC_s002
                                 + 0.917033 w_k1_KO_s002
                                 + 1.06967 w_k1_MAT_s002
                                 + 0.793945 w_k1_MSFT_s002
                                 + 0.858382 w_k1_PG_s002 >= 1
 z_k1_s0022_constraint:          var_1 + z_k1_s0022 + 1.14065 w_k1_ADM_s002
                                 + 0.7133 w_k1_AXP_s002
                                 + 0.959499 w_k1_DIS_s002
                                 + 1.01369 w_k1_FDX_s002
                                 + 0.73088 w_k1_IBM_s002
                                 + 0.888164 w_k1_INTC_s002
                                 + 0.785592 w_k1_KO_s002
                                 + 0.618701 w_k1_MAT_s002
                                 + 1.05395 w_k1_MSFT_s002
                                 + 0.939329 w_k1_PG_s002 >= 1
 z_k1_s0023_constraint:          var_1 + z_k1_s0023 + 1.45928 w_k1_ADM_s002
                                 + 1.42484 w_k1_AXP_s002
                                 + 1.37653 w_k1_DIS_s002
                                 + 1.37748 w_k1_FDX_s002
                                 + 1.13788 w_k1_IBM_s002
                                 + 1.23055 w_k1_INTC_s002
                                 + 1.20199 w_k1_KO_s002 + 1.31321 w_k1_MAT_s002
                                 + 1.26361 w_k1_MSFT_s002
                                 + 1.22479 w_k1_PG_s002 >= 1
 z_k1_s0024_constraint:          var_1 + z_k1_s0024 + 1.13912 w_k1_ADM_s002
                                 + 1.11275 w_k1_AXP_s002
                                 + 1.10189 w_k1_DIS_s002
                                 + 1.03533 w_k1_FDX_s002
                                 + 1.63571 w_k1_IBM_s002
                                 + 1.01885 w_k1_INTC_s002 + 1.2449 w_k1_KO_s002
                                 + 1.43843 w_k1_MAT_s002
                                 + 0.97117 w_k1_MSFT_s002
                                 + 1.13377 w_k1_PG_s002 >= 1
 z_k1_s0025_constraint:          var_1 + z_k1_s0025 + 1.29754 w_k1_ADM_s002
                                 + 1.12585 w_k1_AXP_s002
                                 + 1.09994 w_k1_DIS_s002
                                 + 1.13347 w_k1_FDX_s002
                                 + 1.03777 w_k1_IBM_s002
                                 + 0.939041 w_k1_INTC_s002
                                 + 1.04425 w_k1_KO_s002 + 1.00281 w_k1_MAT_s002
                                 + 0.938132 w_k1_MSFT_s002
                                 + 1.12201 w_k1_PG_s002 >= 1
 z_k1_s0026_constraint:          var_1 + z_k1_s0026 + 1.05359 w_k1_ADM_s002
                                 + 0.637808 w_k1_AXP_s002
                                 + 0.633329 w_k1_DIS_s002
                                 + 0.800423 w_k1_FDX_s002
                                 + 0.586177 w_k1_IBM_s002
                                 + 0.747632 w_k1_INTC_s002
                                 + 0.672959 w_k1_KO_s002
                                 + 0.488137 w_k1_MAT_s002
                                 + 0.770321 w_k1_MSFT_s002
                                 + 0.794474 w_k1_PG_s002 >= 1
 z_k1_s0027_constraint:          var_1 + z_k1_s0027 + 1.2355 w_k1_ADM_s002
                                 + 1.20831 w_k1_AXP_s002
                                 + 0.771576 w_k1_DIS_s002
                                 + 1.14741 w_k1_FDX_s002
                                 + 0.690397 w_k1_IBM_s002
                                 + 0.971211 w_k1_INTC_s002
                                 + 0.728342 w_k1_KO_s002
                                 + 0.754018 w_k1_MAT_s002
                                 + 1.02154 w_k1_MSFT_s002
                                 + 0.96873 w_k1_PG_s002 >= 1
 z_k1_s0030_constraint:          var_1 + z_k1_s0030 + 1.11666 w_k1_ADM_s003
                                 + 1.20174 w_k1_AXP_s003
                                 + 1.32766 w_k1_DIS_s003
                                 + 1.31185 w_k1_FDX_s003
                                 + 0.814773 w_k1_IBM_s003
                                 + 0.931173 w_k1_INTC_s003
                                 + 1.07553 w_k1_KO_s003
                                 + 0.767924 w_k1_MAT_s003
                                 + 1.11621 w_k1_MSFT_s003
                                 + 1.22652 w_k1_PG_s003 >= 1
 z_k1_s0031_constraint:          var_1 + z_k1_s0031 + 1.25285 w_k1_ADM_s003
                                 + 1.14998 w_k1_AXP_s003
                                 + 1.04152 w_k1_DIS_s003
                                 + 1.06514 w_k1_FDX_s003
                                 + 1.13014 w_k1_IBM_s003
                                 + 1.06275 w_k1_INTC_s003
                                 + 1.13104 w_k1_KO_s003 + 1.04936 w_k1_MAT_s003
                                 + 0.951725 w_k1_MSFT_s003
                                 + 1.0893 w_k1_PG_s003 >= 1
 z_k1_s0032_constraint:          var_1 + z_k1_s0032 + 1.28656 w_k1_ADM_s003
                                 + 0.99449 w_k1_AXP_s003
                                 + 1.31386 w_k1_DIS_s003
                                 + 1.14828 w_k1_FDX_s003
                                 + 1.23266 w_k1_IBM_s003
                                 + 0.94486 w_k1_INTC_s003
                                 + 1.10565 w_k1_KO_s003
                                 + 0.849202 w_k1_MAT_s003
                                 + 1.05957 w_k1_MSFT_s003
                                 + 1.05508 w_k1_PG_s003 >= 1
 z_k1_s0033_constraint:          var_1 + z_k1_s0033 + 0.718976 w_k1_ADM_s003
                                 + 0.818301 w_k1_AXP_s003
                                 + 0.685245 w_k1_DIS_s003
                                 + 0.794818 w_k1_FDX_s003
                                 + 0.91449 w_k1_IBM_s003
                                 + 0.762874 w_k1_INTC_s003
                                 + 0.761558 w_k1_KO_s003
                                 + 0.992521 w_k1_MAT_s003
                                 + 0.821651 w_k1_MSFT_s003
                                 + 0.879624 w_k1_PG_s003 >= 1
 z_k1_s0034_constraint:          var_1 + z_k1_s0034 + 1.32147 w_k1_ADM_s003
                                 + 1.4361 w_k1_AXP_s003 + 1.40001 w_k1_DIS_s003
                                 + 1.20784 w_k1_FDX_s003
                                 + 1.61549 w_k1_IBM_s003
                                 + 1.14629 w_k1_INTC_s003
                                 + 1.43363 w_k1_KO_s003 + 1.378 w_k1_MAT_s003
                                 + 1.12175 w_k1_MSFT_s003
                                 + 1.23124 w_k1_PG_s003 >= 1
 z_k1_s0035_constraint:          var_1 + z_k1_s0035 + 0.980827 w_k1_ADM_s003
                                 + 1.00881 w_k1_AXP_s003
                                 + 0.911081 w_k1_DIS_s003
                                 + 0.91536 w_k1_FDX_s003
                                 + 1.05984 w_k1_IBM_s003
                                 + 0.697513 w_k1_INTC_s003
                                 + 1.04565 w_k1_KO_s003 + 1.07087 w_k1_MAT_s003
                                 + 0.795343 w_k1_MSFT_s003
                                 + 0.95384 w_k1_PG_s003 >= 1
 z_k1_s0036_constraint:          var_1 + z_k1_s0036 + 0.973172 w_k1_ADM_s003
                                 + 1.22463 w_k1_AXP_s003
                                 + 0.940487 w_k1_DIS_s003
                                 + 0.932178 w_k1_FDX_s003
                                 + 1.40155 w_k1_IBM_s003
                                 + 0.939442 w_k1_INTC_s003
                                 + 1.17678 w_k1_KO_s003 + 1.44912 w_k1_MAT_s003
                                 + 0.94428 w_k1_MSFT_s003 + 1.1462 w_k1_PG_s003
                                 >= 1
 z_k1_s0037_constraint:          var_1 + z_k1_s0037 + 1.03894 w_k1_ADM_s003
                                 + 0.710797 w_k1_AXP_s003
                                 + 0.70287 w_k1_DIS_s003
                                 + 0.903838 w_k1_FDX_s003
                                 + 0.625146 w_k1_IBM_s003
                                 + 0.715868 w_k1_INTC_s003
                                 + 0.778289 w_k1_KO_s003
                                 + 0.623792 w_k1_MAT_s003
                                 + 0.878774 w_k1_MSFT_s003
                                 + 0.857492 w_k1_PG_s003 >= 1
 z_k1_s0040_constraint:          var_1 + z_k1_s0040 + 1.01311 w_k1_ADM_s004
                                 + 1.15486 w_k1_AXP_s004
                                 + 1.13888 w_k1_DIS_s004
                                 + 1.03005 w_k1_FDX_s004
                                 + 1.69461 w_k1_IBM_s004
                                 + 1.08812 w_k1_INTC_s004
                                 + 1.36353 w_k1_KO_s004 + 2.22759 w_k1_MAT_s004
                                 + 1.11596 w_k1_MSFT_s004
                                 + 1.24084 w_k1_PG_s004 >= 1
 z_k1_s0041_constraint:          var_1 + z_k1_s0041 + 0.937351 w_k1_ADM_s004
                                 + 0.729573 w_k1_AXP_s004
                                 + 0.554044 w_k1_DIS_s004
                                 + 0.848922 w_k1_FDX_s004
                                 + 0.623525 w_k1_IBM_s004
                                 + 1.01679 w_k1_INTC_s004
                                 + 0.757474 w_k1_KO_s004
                                 + 0.682534 w_k1_MAT_s004
                                 + 0.908513 w_k1_MSFT_s004
                                 + 0.941464 w_k1_PG_s004 >= 1
 z_k1_s0042_constraint:          var_1 + z_k1_s0042 + 1.01877 w_k1_ADM_s004
                                 + 0.89993 w_k1_AXP_s004
                                 + 1.01509 w_k1_DIS_s004
                                 + 1.06878 w_k1_FDX_s004
                                 + 0.859453 w_k1_IBM_s004
                                 + 1.06779 w_k1_INTC_s004
                                 + 0.910841 w_k1_KO_s004
                                 + 0.812515 w_k1_MAT_s004
                                 + 1.06486 w_k1_MSFT_s004
                                 + 1.03001 w_k1_PG_s004 >= 1
 z_k1_s0043_constraint:          var_1 + z_k1_s0043 + 0.856412 w_k1_ADM_s004
                                 + 0.946036 w_k1_AXP_s004
                                 + 0.642135 w_k1_DIS_s004
                                 + 0.898609 w_k1_FDX_s004
                                 + 0.759071 w_k1_IBM_s004
                                 + 0.946975 w_k1_INTC_s004
                                 + 0.765654 w_k1_KO_s004
                                 + 0.82502 w_k1_MAT_s004
                                 + 0.94067 w_k1_MSFT_s004
                                 + 0.910395 w_k1_PG_s004 >= 1
 z_k1_s0044_constraint:          var_1 + z_k1_s0044 + 1.05153 w_k1_ADM_s004
                                 + 1.43674 w_k1_AXP_s004
                                 + 1.49654 w_k1_DIS_s004
                                 + 1.37557 w_k1_FDX_s004
                                 + 1.14879 w_k1_IBM_s004
                                 + 1.63179 w_k1_INTC_s004
                                 + 1.34257 w_k1_KO_s004 + 1.77376 w_k1_MAT_s004
                                 + 1.32843 w_k1_MSFT_s004
                                 + 1.27946 w_k1_PG_s004 >= 1
 z_k1_s0045_constraint:          var_1 + z_k1_s0045 + 0.746686 w_k1_ADM_s004
                                 + 0.542696 w_k1_AXP_s004
                                 + 0.59658 w_k1_DIS_s004
                                 + 0.741092 w_k1_FDX_s004
                                 + 0.655405 w_k1_IBM_s004
                                 + 0.784201 w_k1_INTC_s004
                                 + 0.727728 w_k1_KO_s004
                                 + 0.591297 w_k1_MAT_s004
                                 + 0.888918 w_k1_MSFT_s004
                                 + 0.8319 w_k1_PG_s004 >= 1
 z_k1_s0046_constraint:          var_1 + z_k1_s0046 + 1.03397 w_k1_ADM_s004
                                 + 1.31369 w_k1_AXP_s004
                                 + 0.922451 w_k1_DIS_s004
                                 + 1.03708 w_k1_FDX_s004
                                 + 1.10032 w_k1_IBM_s004
                                 + 0.947517 w_k1_INTC_s004
                                 + 0.93075 w_k1_KO_s004 + 1.2059 w_k1_MAT_s004
                                 + 1.03112 w_k1_MSFT_s004
                                 + 1.12763 w_k1_PG_s004 >= 1
 z_k1_s0047_constraint:          var_1 + z_k1_s0047 + 0.792785 w_k1_ADM_s004
                                 + 0.896169 w_k1_AXP_s004
                                 + 0.704917 w_k1_DIS_s004
                                 + 0.828817 w_k1_FDX_s004
                                 + 1.16096 w_k1_IBM_s004
                                 + 1.04407 w_k1_INTC_s004
                                 + 0.928854 w_k1_KO_s004
                                 + 1.08452 w_k1_MAT_s004 + 0.968 w_k1_MSFT_s004
                                 + 0.907064 w_k1_PG_s004 >= 1
 z_k1_s0050_constraint:          var_1 + z_k1_s0050 + 1.45497 w_k1_ADM_s005
                                 + 1.55083 w_k1_AXP_s005 + 1.6685 w_k1_DIS_s005
                                 + 1.20843 w_k1_FDX_s005
                                 + 1.20139 w_k1_IBM_s005
                                 + 1.36519 w_k1_INTC_s005 + 1.3922 w_k1_KO_s005
                                 + 1.29855 w_k1_MAT_s005
                                 + 1.23338 w_k1_MSFT_s005
                                 + 1.23548 w_k1_PG_s005 >= 1
 z_k1_s0051_constraint:          var_1 + z_k1_s0051 + 1.23094 w_k1_ADM_s005
                                 + 1.22101 w_k1_AXP_s005
                                 + 1.38777 w_k1_DIS_s005
                                 + 1.11453 w_k1_FDX_s005
                                 + 1.51965 w_k1_IBM_s005
                                 + 0.997867 w_k1_INTC_s005
                                 + 1.49874 w_k1_KO_s005 + 1.96752 w_k1_MAT_s005
                                 + 1.2173 w_k1_MSFT_s005 + 1.1876 w_k1_PG_s005
                                 >= 1
 z_k1_s0052_constraint:          var_1 + z_k1_s0052 + 0.961488 w_k1_ADM_s005
                                 + 0.890537 w_k1_AXP_s005
                                 + 1.03089 w_k1_DIS_s005
                                 + 0.999089 w_k1_FDX_s005
                                 + 1.08495 w_k1_IBM_s005
                                 + 0.999588 w_k1_INTC_s005
                                 + 1.19134 w_k1_KO_s005 + 1.11232 w_k1_MAT_s005
                                 + 1.00387 w_k1_MSFT_s005
                                 + 0.973593 w_k1_PG_s005 >= 1
 z_k1_s0053_constraint:          var_1 + z_k1_s0053 + 1.17228 w_k1_ADM_s005
                                 + 1.10475 w_k1_AXP_s005
                                 + 1.34177 w_k1_DIS_s005
                                 + 1.14885 w_k1_FDX_s005
                                 + 0.889958 w_k1_IBM_s005
                                 + 0.95401 w_k1_INTC_s005
                                 + 1.08864 w_k1_KO_s005 + 1.10331 w_k1_MAT_s005
                                 + 1.1302 w_k1_MSFT_s005 + 1.11194 w_k1_PG_s005
                                 >= 1
 z_k1_s0054_constraint:          var_1 + z_k1_s0054 + 1.00818 w_k1_ADM_s005
                                 + 0.830539 w_k1_AXP_s005
                                 + 0.912685 w_k1_DIS_s005
                                 + 0.982872 w_k1_FDX_s005
                                 + 0.689669 w_k1_IBM_s005
                                 + 0.981096 w_k1_INTC_s005
                                 + 0.999955 w_k1_KO_s005
                                 + 0.825978 w_k1_MAT_s005
                                 + 0.94631 w_k1_MSFT_s005
                                 + 0.933994 w_k1_PG_s005 >= 1
 z_k1_s0055_constraint:          var_1 + z_k1_s0055 + 0.847676 w_k1_ADM_s005
                                 + 0.744896 w_k1_AXP_s005
                                 + 0.71131 w_k1_DIS_s005
                                 + 0.695792 w_k1_FDX_s005
                                 + 0.957963 w_k1_IBM_s005
                                 + 0.944123 w_k1_INTC_s005
                                 + 1.09417 w_k1_KO_s005 + 1.01979 w_k1_MAT_s005
                                 + 0.835207 w_k1_MSFT_s005
                                 + 0.838667 w_k1_PG_s005 >= 1
 z_k1_s0056_constraint:          var_1 + z_k1_s0056 + 1.00014 w_k1_ADM_s005
                                 + 1.26985 w_k1_AXP_s005
                                 + 1.15186 w_k1_DIS_s005
                                 + 1.09446 w_k1_FDX_s005
                                 + 1.04114 w_k1_IBM_s005
                                 + 1.08683 w_k1_INTC_s005
                                 + 1.20318 w_k1_KO_s005 + 1.79233 w_k1_MAT_s005
                                 + 1.12589 w_k1_MSFT_s005
                                 + 1.11981 w_k1_PG_s005 >= 1
 z_k1_s0057_constraint:          var_1 + z_k1_s0057 + 1.02828 w_k1_ADM_s005
                                 + 1.18144 w_k1_AXP_s005
                                 + 0.741065 w_k1_DIS_s005
                                 + 0.902182 w_k1_FDX_s005
                                 + 1.26538 w_k1_IBM_s005
                                 + 1.06143 w_k1_INTC_s005
                                 + 1.08093 w_k1_KO_s005 + 1.33133 w_k1_MAT_s005
                                 + 0.860088 w_k1_MSFT_s005
                                 + 1.03296 w_k1_PG_s005 >= 1
 z_k1_s0060_constraint:          var_1 + z_k1_s0060 + 0.941947 w_k1_ADM_s006
                                 + 0.873493 w_k1_AXP_s006
                                 + 0.936997 w_k1_DIS_s006
                                 + 1.01516 w_k1_FDX_s006
                                 + 0.670408 w_k1_IBM_s006
                                 + 0.974072 w_k1_INTC_s006
                                 + 0.836146 w_k1_KO_s006
                                 + 0.788751 w_k1_MAT_s006
                                 + 1.01576 w_k1_MSFT_s006
                                 + 0.920011 w_k1_PG_s006 >= 1
 z_k1_s0061_constraint:          var_1 + z_k1_s0061 + 1.45861 w_k1_ADM_s006
                                 + 1.4794 w_k1_AXP_s006 + 1.67238 w_k1_DIS_s006
                                 + 1.4858 w_k1_FDX_s006 + 1.04858 w_k1_IBM_s006
                                 + 1.71334 w_k1_INTC_s006
                                 + 1.22625 w_k1_KO_s006 + 1.42984 w_k1_MAT_s006
                                 + 1.71086 w_k1_MSFT_s006
                                 + 1.19416 w_k1_PG_s006 >= 1
 z_k1_s0062_constraint:          var_1 + z_k1_s0062 + 1.50635 w_k1_ADM_s006
                                 + 1.70245 w_k1_AXP_s006
                                 + 1.69402 w_k1_DIS_s006
                                 + 1.50362 w_k1_FDX_s006
                                 + 1.61507 w_k1_IBM_s006
                                 + 1.33614 w_k1_INTC_s006
                                 + 1.26986 w_k1_KO_s006 + 1.86713 w_k1_MAT_s006
                                 + 1.20675 w_k1_MSFT_s006
                                 + 1.16921 w_k1_PG_s006 >= 1
 z_k1_s0063_constraint:          var_1 + z_k1_s0063 + 1.06879 w_k1_ADM_s006
                                 + 1.06018 w_k1_AXP_s006
                                 + 1.18658 w_k1_DIS_s006
                                 + 1.13614 w_k1_FDX_s006
                                 + 1.02246 w_k1_IBM_s006
                                 + 0.973822 w_k1_INTC_s006
                                 + 0.987874 w_k1_KO_s006
                                 + 1.06342 w_k1_MAT_s006
                                 + 1.11377 w_k1_MSFT_s006
                                 + 1.06932 w_k1_PG_s006 >= 1
 z_k1_s0064_constraint:          var_1 + z_k1_s0064 + 0.784193 w_k1_ADM_s006
                                 + 0.771107 w_k1_AXP_s006
                                 + 0.774552 w_k1_DIS_s006
                                 + 0.741653 w_k1_FDX_s006
                                 + 1.38897 w_k1_IBM_s006
                                 + 1.06826 w_k1_INTC_s006 + 1.1034 w_k1_KO_s006
                                 + 1.50399 w_k1_MAT_s006
                                 + 0.9623 w_k1_MSFT_s006
                                 + 0.935452 w_k1_PG_s006 >= 1
 z_k1_s0065_constraint:          var_1 + z_k1_s0065 + 0.912866 w_k1_ADM_s006
                                 + 0.97944 w_k1_AXP_s006
                                 + 1.09802 w_k1_DIS_s006
                                 + 0.866518 w_k1_FDX_s006
                                 + 1.27802 w_k1_IBM_s006
                                 + 1.12809 w_k1_INTC_s006
                                 + 1.14044 w_k1_KO_s006 + 1.32518 w_k1_MAT_s006
                                 + 1.07287 w_k1_MSFT_s006
                                 + 0.960168 w_k1_PG_s006 >= 1
 z_k1_s0066_constraint:          var_1 + z_k1_s0066 + 0.743068 w_k1_ADM_s006
                                 + 0.589198 w_k1_AXP_s006
                                 + 0.604995 w_k1_DIS_s006
                                 + 0.694129 w_k1_FDX_s006
                                 + 0.943257 w_k1_IBM_s006
                                 + 1.06711 w_k1_INTC_s006
                                 + 0.897641 w_k1_KO_s006
                                 + 0.837739 w_k1_MAT_s006
                                 + 0.921546 w_k1_MSFT_s006
                                 + 0.854252 w_k1_PG_s006 >= 1
 z_k1_s0067_constraint:          var_1 + z_k1_s0067 + 0.842502 w_k1_ADM_s006
                                 + 0.48371 w_k1_AXP_s006
                                 + 0.547569 w_k1_DIS_s006
                                 + 0.644026 w_k1_FDX_s006
                                 + 0.710232 w_k1_IBM_s006
                                 + 0.701152 w_k1_INTC_s006
                                 + 0.651802 w_k1_KO_s006
                                 + 0.52945 w_k1_MAT_s006
                                 + 0.718729 w_k1_MSFT_s006
                                 + 0.654784 w_k1_PG_s006 >= 1
 z_k1_s0070_constraint:          var_1 + z_k1_s0070 + 1.22416 w_k1_ADM_s007
                                 + 1.16418 w_k1_AXP_s007
                                 + 1.65867 w_k1_DIS_s007
                                 + 1.19337 w_k1_FDX_s007
                                 + 1.51239 w_k1_IBM_s007
                                 + 1.15265 w_k1_INTC_s007
                                 + 1.27903 w_k1_KO_s007 + 1.83397 w_k1_MAT_s007
                                 + 1.10441 w_k1_MSFT_s007
                                 + 1.27525 w_k1_PG_s007 >= 1
 z_k1_s0071_constraint:          var_1 + z_k1_s0071 + 1.13472 w_k1_ADM_s007
                                 + 0.926997 w_k1_AXP_s007
                                 + 0.898785 w_k1_DIS_s007
                                 + 1.03152 w_k1_FDX_s007
                                 + 0.750376 w_k1_IBM_s007
                                 + 1.09943 w_k1_INTC_s007
                                 + 0.930532 w_k1_KO_s007
                                 + 0.625323 w_k1_MAT_s007
                                 + 0.982853 w_k1_MSFT_s007
                                 + 0.960411 w_k1_PG_s007 >= 1
 z_k1_s0072_constraint:          var_1 + z_k1_s0072 + 1.12759 w_k1_ADM_s007
                                 + 0.899283 w_k1_AXP_s007
                                 + 0.944247 w_k1_DIS_s007
                                 + 0.865119 w_k1_FDX_s007
                                 + 1.63975 w_k1_IBM_s007
                                 + 0.811329 w_k1_INTC_s007
                                 + 1.10535 w_k1_KO_s007 + 1.10913 w_k1_MAT_s007
                                 + 0.889431 w_k1_MSFT_s007
                                 + 1.01098 w_k1_PG_s007 >= 1
 z_k1_s0073_constraint:          var_1 + z_k1_s0073 + 1.09109 w_k1_ADM_s007
                                 + 1.02714 w_k1_AXP_s007
                                 + 0.86168 w_k1_DIS_s007
                                 + 0.825019 w_k1_FDX_s007
                                 + 1.11799 w_k1_IBM_s007
                                 + 0.94825 w_k1_INTC_s007
                                 + 0.791497 w_k1_KO_s007
                                 + 1.12888 w_k1_MAT_s007
                                 + 0.846614 w_k1_MSFT_s007
                                 + 0.951731 w_k1_PG_s007 >= 1
 z_k1_s0074_constraint:          var_1 + z_k1_s0074 + 1.40466 w_k1_ADM_s007
                                 + 1.45896 w_k1_AXP_s007
                                 + 1.60781 w_k1_DIS_s007
                                 + 1.41768 w_k1_FDX_s007
                                 + 1.08138 w_k1_IBM_s007
                                 + 1.47917 w_k1_INTC_s007
                                 + 1.14579 w_k1_KO_s007 + 1.04155 w_k1_MAT_s007
                                 + 1.38998 w_k1_MSFT_s007
                                 + 1.20164 w_k1_PG_s007 >= 1
 z_k1_s0075_constraint:          var_1 + z_k1_s0075 + 1.18935 w_k1_ADM_s007
                                 + 1.08785 w_k1_AXP_s007
                                 + 1.18779 w_k1_DIS_s007
                                 + 1.13719 w_k1_FDX_s007
                                 + 1.04604 w_k1_IBM_s007
                                 + 1.01951 w_k1_INTC_s007
                                 + 1.03618 w_k1_KO_s007 + 1.06266 w_k1_MAT_s007
                                 + 0.997534 w_k1_MSFT_s007
                                 + 0.986001 w_k1_PG_s007 >= 1
 z_k1_s0076_constraint:          var_1 + z_k1_s0076 + 0.810821 w_k1_ADM_s007
                                 + 0.792053 w_k1_AXP_s007
                                 + 0.629312 w_k1_DIS_s007
                                 + 0.803518 w_k1_FDX_s007
                                 + 0.798097 w_k1_IBM_s007
                                 + 0.833657 w_k1_INTC_s007
                                 + 0.635475 w_k1_KO_s007
                                 + 0.843805 w_k1_MAT_s007
                                 + 0.743461 w_k1_MSFT_s007
                                 + 0.824683 w_k1_PG_s007 >= 1
 z_k1_s0077_constraint:          var_1 + z_k1_s0077 + 1.43102 w_k1_ADM_s007
                                 + 1.37608 w_k1_AXP_s007
                                 + 1.06794 w_k1_DIS_s007
                                 + 1.09636 w_k1_FDX_s007
                                 + 1.18127 w_k1_IBM_s007
                                 + 1.04116 w_k1_INTC_s007
                                 + 1.07715 w_k1_KO_s007 + 1.34855 w_k1_MAT_s007
                                 + 0.977639 w_k1_MSFT_s007
                                 + 1.20618 w_k1_PG_s007 >= 1
 z_k1_s0100_constraint:          var_1 + z_k1_s0100 + 0.780408 w_k1_ADM_s010
                                 + 0.93299 w_k1_AXP_s010
                                 + 0.991935 w_k1_DIS_s010
                                 + 0.967078 w_k1_FDX_s010
                                 + 1.00072 w_k1_IBM_s010
                                 + 1.05912 w_k1_INTC_s010
                                 + 1.10822 w_k1_KO_s010
                                 + 0.806321 w_k1_MAT_s010
                                 + 0.979526 w_k1_MSFT_s010
                                 + 0.994464 w_k1_PG_s010 >= 1
 z_k1_s0101_constraint:          var_1 + z_k1_s0101 + 0.965419 w_k1_ADM_s010
                                 + 1.06413 w_k1_AXP_s010
                                 + 1.37319 w_k1_DIS_s010
                                 + 1.11821 w_k1_FDX_s010
                                 + 1.25111 w_k1_IBM_s010
                                 + 1.11326 w_k1_INTC_s010
                                 + 1.31549 w_k1_KO_s010
                                 + 0.979022 w_k1_MAT_s010
                                 + 1.17324 w_k1_MSFT_s010
                                 + 1.11034 w_k1_PG_s010 >= 1
 z_k1_s0102_constraint:          var_1 + z_k1_s0102 + 1.03011 w_k1_ADM_s010
                                 + 0.822106 w_k1_AXP_s010
                                 + 1.03918 w_k1_DIS_s010
                                 + 1.01548 w_k1_FDX_s010
                                 + 0.676028 w_k1_IBM_s010
                                 + 0.926117 w_k1_INTC_s010
                                 + 0.786734 w_k1_KO_s010
                                 + 0.430114 w_k1_MAT_s010
                                 + 0.949595 w_k1_MSFT_s010
                                 + 0.911375 w_k1_PG_s010 >= 1
 z_k1_s0103_constraint:          var_1 + z_k1_s0103 + 0.792892 w_k1_ADM_s010
                                 + 0.804095 w_k1_AXP_s010
                                 + 0.699317 w_k1_DIS_s010
                                 + 0.774753 w_k1_FDX_s010
                                 + 0.834916 w_k1_IBM_s010
                                 + 0.762616 w_k1_INTC_s010
                                 + 0.768853 w_k1_KO_s010
                                 + 0.885949 w_k1_MAT_s010
                                 + 0.717068 w_k1_MSFT_s010
                                 + 0.814318 w_k1_PG_s010 >= 1
 z_k1_s0104_constraint:          var_1 + z_k1_s0104 + 0.738049 w_k1_ADM_s010
                                 + 0.580262 w_k1_AXP_s010
                                 + 0.882041 w_k1_DIS_s010
                                 + 0.761691 w_k1_FDX_s010
                                 + 0.826049 w_k1_IBM_s010
                                 + 0.806252 w_k1_INTC_s010
                                 + 0.877468 w_k1_KO_s010
                                 + 0.650277 w_k1_MAT_s010
                                 + 0.850886 w_k1_MSFT_s010
                                 + 0.813936 w_k1_PG_s010 >= 1
 z_k1_s0105_constraint:          var_1 + z_k1_s0105 + 0.961502 w_k1_ADM_s010
                                 + 1.27328 w_k1_AXP_s010
                                 + 0.829365 w_k1_DIS_s010
                                 + 0.985585 w_k1_FDX_s010
                                 + 1.08758 w_k1_IBM_s010
                                 + 0.808294 w_k1_INTC_s010
                                 + 0.916862 w_k1_KO_s010
                                 + 1.09565 w_k1_MAT_s010
                                 + 0.764486 w_k1_MSFT_s010
                                 + 1.04967 w_k1_PG_s010 >= 1
 z_k1_s0106_constraint:          var_1 + z_k1_s0106 + 1.0632 w_k1_ADM_s010
                                 + 1.27766 w_k1_AXP_s010
                                 + 1.27923 w_k1_DIS_s010
                                 + 1.13308 w_k1_FDX_s010
                                 + 1.50982 w_k1_IBM_s010
                                 + 1.11276 w_k1_INTC_s010
                                 + 1.14289 w_k1_KO_s010 + 1.12997 w_k1_MAT_s010
                                 + 1.00136 w_k1_MSFT_s010
                                 + 1.01161 w_k1_PG_s010 >= 1
 z_k1_s0107_constraint:          var_1 + z_k1_s0107 + 1.34676 w_k1_ADM_s010
                                 + 1.19914 w_k1_AXP_s010
                                 + 1.36054 w_k1_DIS_s010
                                 + 1.28366 w_k1_FDX_s010
                                 + 0.974685 w_k1_IBM_s010
                                 + 1.274 w_k1_INTC_s010 + 1.18086 w_k1_KO_s010
                                 + 0.62466 w_k1_MAT_s010
                                 + 1.12846 w_k1_MSFT_s010
                                 + 1.13004 w_k1_PG_s010 >= 1
 z_k1_s0110_constraint:          var_1 + z_k1_s0110 + 0.777423 w_k1_ADM_s011
                                 + 0.94234 w_k1_AXP_s011
                                 + 0.753392 w_k1_DIS_s011
                                 + 0.822865 w_k1_FDX_s011
                                 + 1.10346 w_k1_IBM_s011
                                 + 1.0203 w_k1_INTC_s011
                                 + 0.987679 w_k1_KO_s011
                                 + 1.02307 w_k1_MAT_s011
                                 + 0.850949 w_k1_MSFT_s011
                                 + 0.929436 w_k1_PG_s011 >= 1
 z_k1_s0111_constraint:          var_1 + z_k1_s0111 + 0.937946 w_k1_ADM_s011
                                 + 1.2177 w_k1_AXP_s011 + 1.24823 w_k1_DIS_s011
                                 + 1.03414 w_k1_FDX_s011
                                 + 1.42519 w_k1_IBM_s011
                                 + 1.07575 w_k1_INTC_s011
                                 + 1.40578 w_k1_KO_s011 + 1.55079 w_k1_MAT_s011
                                 + 1.06298 w_k1_MSFT_s011
                                 + 1.09343 w_k1_PG_s011 >= 1
 z_k1_s0112_constraint:          var_1 + z_k1_s0112 + 1.15387 w_k1_ADM_s011
                                 + 1.20617 w_k1_AXP_s011 + 1.0842 w_k1_DIS_s011
                                 + 1.08889 w_k1_FDX_s011
                                 + 1.07707 w_k1_IBM_s011
                                 + 1.21352 w_k1_INTC_s011
                                 + 1.17862 w_k1_KO_s011
                                 + 0.782595 w_k1_MAT_s011
                                 + 1.13981 w_k1_MSFT_s011
                                 + 1.09016 w_k1_PG_s011 >= 1
 z_k1_s0113_constraint:          var_1 + z_k1_s0113 + 0.828237 w_k1_ADM_s011
                                 + 0.695475 w_k1_AXP_s011
                                 + 0.612862 w_k1_DIS_s011
                                 + 0.77163 w_k1_FDX_s011
                                 + 0.734531 w_k1_IBM_s011
                                 + 0.926278 w_k1_INTC_s011
                                 + 0.746879 w_k1_KO_s011
                                 + 0.528991 w_k1_MAT_s011
                                 + 0.688205 w_k1_MSFT_s011
                                 + 0.743586 w_k1_PG_s011 >= 1
 z_k1_s0114_constraint:          var_1 + z_k1_s0114 + 1.26341 w_k1_ADM_s011
                                 + 1.48911 w_k1_AXP_s011
                                 + 1.35189 w_k1_DIS_s011 + 1.3178 w_k1_FDX_s011
                                 + 1.01346 w_k1_IBM_s011
                                 + 1.45758 w_k1_INTC_s011
                                 + 1.11275 w_k1_KO_s011 + 1.14605 w_k1_MAT_s011
                                 + 1.12501 w_k1_MSFT_s011
                                 + 0.999316 w_k1_PG_s011 >= 1
 z_k1_s0115_constraint:          var_1 + z_k1_s0115 + 1.09064 w_k1_ADM_s011
                                 + 1.30979 w_k1_AXP_s011
                                 + 0.940782 w_k1_DIS_s011
                                 + 1.14142 w_k1_FDX_s011
                                 + 0.815645 w_k1_IBM_s011
                                 + 1.18938 w_k1_INTC_s011
                                 + 0.984811 w_k1_KO_s011
                                 + 0.782751 w_k1_MAT_s011
                                 + 1.0385 w_k1_MSFT_s011
                                 + 0.934104 w_k1_PG_s011 >= 1
 z_k1_s0116_constraint:          var_1 + z_k1_s0116 + 1.55306 w_k1_ADM_s011
                                 + 1.62053 w_k1_AXP_s011
                                 + 1.94087 w_k1_DIS_s011
                                 + 1.54406 w_k1_FDX_s011
                                 + 1.25836 w_k1_IBM_s011
                                 + 1.35495 w_k1_INTC_s011
                                 + 1.50557 w_k1_KO_s011
                                 + 0.963508 w_k1_MAT_s011
                                 + 1.23446 w_k1_MSFT_s011
                                 + 1.20723 w_k1_PG_s011 >= 1
 z_k1_s0117_constraint:          var_1 + z_k1_s0117 + 0.785315 w_k1_ADM_s011
                                 + 0.973261 w_k1_AXP_s011
                                 + 1.01148 w_k1_DIS_s011
                                 + 0.996578 w_k1_FDX_s011
                                 + 0.811629 w_k1_IBM_s011
                                 + 1.2009 w_k1_INTC_s011
                                 + 0.821898 w_k1_KO_s011
                                 + 0.87282 w_k1_MAT_s011
                                 + 1.10622 w_k1_MSFT_s011
                                 + 0.968446 w_k1_PG_s011 >= 1
 z_k1_s0120_constraint:          var_1 + z_k1_s0120 + 1.41773 w_k1_ADM_s012
                                 + 1.17333 w_k1_AXP_s012
                                 + 1.36849 w_k1_DIS_s012
                                 + 1.26225 w_k1_FDX_s012 + 0.7232 w_k1_IBM_s012
                                 + 0.855272 w_k1_INTC_s012
                                 + 0.917762 w_k1_KO_s012
                                 + 0.61989 w_k1_MAT_s012
                                 + 1.2042 w_k1_MSFT_s012 + 1.03898 w_k1_PG_s012
                                 >= 1
 z_k1_s0121_constraint:          var_1 + z_k1_s0121 + 1.33163 w_k1_ADM_s012
                                 + 0.979463 w_k1_AXP_s012
                                 + 1.04623 w_k1_DIS_s012
                                 + 1.04064 w_k1_FDX_s012
                                 + 1.14826 w_k1_IBM_s012
                                 + 0.90275 w_k1_INTC_s012
                                 + 1.30327 w_k1_KO_s012
                                 + 0.905113 w_k1_MAT_s012
                                 + 1.1777 w_k1_MSFT_s012 + 1.07322 w_k1_PG_s012
                                 >= 1
 z_k1_s0122_constraint:          var_1 + z_k1_s0122 + 0.7937 w_k1_ADM_s012
                                 + 0.640452 w_k1_AXP_s012
                                 + 0.691698 w_k1_DIS_s012
                                 + 0.758371 w_k1_FDX_s012
                                 + 1.17021 w_k1_IBM_s012
                                 + 0.833306 w_k1_INTC_s012
                                 + 0.992484 w_k1_KO_s012
                                 + 0.943135 w_k1_MAT_s012
                                 + 0.814539 w_k1_MSFT_s012
                                 + 0.869656 w_k1_PG_s012 >= 1
 z_k1_s0123_constraint:          var_1 + z_k1_s0123 + 1.49471 w_k1_ADM_s012
                                 + 1.21165 w_k1_AXP_s012
                                 + 1.38394 w_k1_DIS_s012
                                 + 1.20637 w_k1_FDX_s012
                                 + 1.63215 w_k1_IBM_s012
                                 + 0.856961 w_k1_INTC_s012
                                 + 1.41091 w_k1_KO_s012 + 1.67498 w_k1_MAT_s012
                                 + 1.13209 w_k1_MSFT_s012 + 1.0886 w_k1_PG_s012
                                 >= 1
 z_k1_s0124_constraint:          var_1 + z_k1_s0124 + 1.62588 w_k1_ADM_s012
                                 + 1.52952 w_k1_AXP_s012
                                 + 1.68597 w_k1_DIS_s012
                                 + 1.62452 w_k1_FDX_s012
                                 + 1.31728 w_k1_IBM_s012
                                 + 1.32329 w_k1_INTC_s012
                                 + 1.58117 w_k1_KO_s012 + 1.39024 w_k1_MAT_s012
                                 + 1.64868 w_k1_MSFT_s012
                                 + 1.34429 w_k1_PG_s012 >= 1
 z_k1_s0125_constraint:          var_1 + z_k1_s0125 + 1.13781 w_k1_ADM_s012
                                 + 1.33696 w_k1_AXP_s012
                                 + 1.09561 w_k1_DIS_s012
                                 + 1.08255 w_k1_FDX_s012
                                 + 0.853336 w_k1_IBM_s012
                                 + 0.888647 w_k1_INTC_s012
                                 + 0.920017 w_k1_KO_s012 + 1.0975 w_k1_MAT_s012
                                 + 1.09404 w_k1_MSFT_s012
                                 + 0.994133 w_k1_PG_s012 >= 1
 z_k1_s0126_constraint:          var_1 + z_k1_s0126 + 0.920102 w_k1_ADM_s012
                                 + 0.687958 w_k1_AXP_s012
                                 + 0.589882 w_k1_DIS_s012
                                 + 0.782194 w_k1_FDX_s012
                                 + 0.72047 w_k1_IBM_s012
                                 + 0.730612 w_k1_INTC_s012
                                 + 0.703378 w_k1_KO_s012
                                 + 0.529742 w_k1_MAT_s012
                                 + 0.79793 w_k1_MSFT_s012
                                 + 0.764105 w_k1_PG_s012 >= 1
 z_k1_s0127_constraint:          var_1 + z_k1_s0127 + 1.15532 w_k1_ADM_s012
                                 + 0.912933 w_k1_AXP_s012
                                 + 1.06091 w_k1_DIS_s012
                                 + 1.16769 w_k1_FDX_s012
                                 + 0.873328 w_k1_IBM_s012
                                 + 0.9586 w_k1_INTC_s012 + 1.00424 w_k1_KO_s012
                                 + 0.621996 w_k1_MAT_s012
                                 + 1.21951 w_k1_MSFT_s012
                                 + 1.00433 w_k1_PG_s012 >= 1
 z_k1_s0130_constraint:          var_1 + z_k1_s0130 + 0.8513 w_k1_ADM_s013
                                 + 0.784493 w_k1_AXP_s013
                                 + 0.692041 w_k1_DIS_s013
                                 + 0.769998 w_k1_FDX_s013
                                 + 0.728893 w_k1_IBM_s013
                                 + 0.902772 w_k1_INTC_s013
                                 + 0.778339 w_k1_KO_s013
                                 + 0.600355 w_k1_MAT_s013
                                 + 0.750938 w_k1_MSFT_s013
                                 + 0.912397 w_k1_PG_s013 >= 1
 z_k1_s0131_constraint:          var_1 + z_k1_s0131 + 0.798648 w_k1_ADM_s013
                                 + 0.80842 w_k1_AXP_s013
                                 + 0.668369 w_k1_DIS_s013
                                 + 0.708646 w_k1_FDX_s013
                                 + 1.36334 w_k1_IBM_s013
                                 + 0.897145 w_k1_INTC_s013
                                 + 0.986362 w_k1_KO_s013
                                 + 1.54876 w_k1_MAT_s013
                                 + 0.869027 w_k1_MSFT_s013
                                 + 1.08011 w_k1_PG_s013 >= 1
 z_k1_s0132_constraint:          var_1 + z_k1_s0132 + 1.39595 w_k1_ADM_s013
                                 + 1.4787 w_k1_AXP_s013 + 1.36196 w_k1_DIS_s013
                                 + 1.32834 w_k1_FDX_s013
                                 + 0.858313 w_k1_IBM_s013
                                 + 1.33323 w_k1_INTC_s013
                                 + 1.02866 w_k1_KO_s013 + 1.24693 w_k1_MAT_s013
                                 + 1.22699 w_k1_MSFT_s013
                                 + 1.07141 w_k1_PG_s013 >= 1
 z_k1_s0133_constraint:          var_1 + z_k1_s0133 + 1.1185 w_k1_ADM_s013
                                 + 0.996865 w_k1_AXP_s013
                                 + 0.969692 w_k1_DIS_s013
                                 + 1.05581 w_k1_FDX_s013
                                 + 0.82962 w_k1_IBM_s013
                                 + 1.01469 w_k1_INTC_s013
                                 + 0.937292 w_k1_KO_s013
                                 + 0.782739 w_k1_MAT_s013
                                 + 1.05037 w_k1_MSFT_s013
                                 + 0.984028 w_k1_PG_s013 >= 1
 z_k1_s0134_constraint:          var_1 + z_k1_s0134 + 1.26685 w_k1_ADM_s013
                                 + 1.36272 w_k1_AXP_s013
                                 + 1.12961 w_k1_DIS_s013
                                 + 0.982513 w_k1_FDX_s013
                                 + 1.56736 w_k1_IBM_s013
                                 + 0.949649 w_k1_INTC_s013
                                 + 1.28712 w_k1_KO_s013 + 1.57196 w_k1_MAT_s013
                                 + 1.00932 w_k1_MSFT_s013
                                 + 1.17735 w_k1_PG_s013 >= 1
 z_k1_s0135_constraint:          var_1 + z_k1_s0135 + 1.0004 w_k1_ADM_s013
                                 + 0.804098 w_k1_AXP_s013
                                 + 0.611526 w_k1_DIS_s013
                                 + 0.757418 w_k1_FDX_s013
                                 + 0.980816 w_k1_IBM_s013
                                 + 0.887955 w_k1_INTC_s013
                                 + 0.885496 w_k1_KO_s013
                                 + 1.15779 w_k1_MAT_s013
                                 + 0.795005 w_k1_MSFT_s013
                                 + 0.863315 w_k1_PG_s013 >= 1
 z_k1_s0136_constraint:          var_1 + z_k1_s0136 + 1.2716 w_k1_ADM_s013
                                 + 1.15326 w_k1_AXP_s013
                                 + 1.21851 w_k1_DIS_s013
                                 + 1.13715 w_k1_FDX_s013
                                 + 1.21311 w_k1_IBM_s013
                                 + 1.09084 w_k1_INTC_s013
                                 + 1.06063 w_k1_KO_s013 + 1.1777 w_k1_MAT_s013
                                 + 1.07066 w_k1_MSFT_s013
                                 + 1.20199 w_k1_PG_s013 >= 1
 z_k1_s0137_constraint:          var_1 + z_k1_s0137 + 1.17085 w_k1_ADM_s013
                                 + 0.99221 w_k1_AXP_s013
                                 + 0.93941 w_k1_DIS_s013
                                 + 0.915535 w_k1_FDX_s013
                                 + 1.19633 w_k1_IBM_s013
                                 + 0.93766 w_k1_INTC_s013
                                 + 0.976501 w_k1_KO_s013
                                 + 1.39203 w_k1_MAT_s013
                                 + 0.901691 w_k1_MSFT_s013
                                 + 1.02226 w_k1_PG_s013 >= 1
 z_k1_s0140_constraint:          var_1 + z_k1_s0140 + 1.0793 w_k1_ADM_s014
                                 + 1.086 w_k1_AXP_s014 + 1.57149 w_k1_DIS_s014
                                 + 1.20837 w_k1_FDX_s014
                                 + 1.22924 w_k1_IBM_s014
                                 + 1.06368 w_k1_INTC_s014
                                 + 1.39325 w_k1_KO_s014 + 1.32335 w_k1_MAT_s014
                                 + 1.27129 w_k1_MSFT_s014
                                 + 1.08317 w_k1_PG_s014 >= 1
 z_k1_s0141_constraint:          var_1 + z_k1_s0141 + 0.774265 w_k1_ADM_s014
                                 + 0.882606 w_k1_AXP_s014
                                 + 0.638532 w_k1_DIS_s014
                                 + 0.859543 w_k1_FDX_s014
                                 + 0.941087 w_k1_IBM_s014
                                 + 0.801683 w_k1_INTC_s014
                                 + 0.726544 w_k1_KO_s014
                                 + 0.934337 w_k1_MAT_s014
                                 + 0.828694 w_k1_MSFT_s014
                                 + 0.765597 w_k1_PG_s014 >= 1
 z_k1_s0142_constraint:          var_1 + z_k1_s0142 + 1.01313 w_k1_ADM_s014
                                 + 1.23037 w_k1_AXP_s014
                                 + 0.978873 w_k1_DIS_s014
                                 + 0.896784 w_k1_FDX_s014
                                 + 1.38498 w_k1_IBM_s014
                                 + 0.824842 w_k1_INTC_s014
                                 + 1.10511 w_k1_KO_s014 + 1.66289 w_k1_MAT_s014
                                 + 0.933541 w_k1_MSFT_s014
                                 + 0.986403 w_k1_PG_s014 >= 1
 z_k1_s0143_constraint:          var_1 + z_k1_s0143 + 0.900062 w_k1_ADM_s014
                                 + 0.892794 w_k1_AXP_s014
                                 + 0.730443 w_k1_DIS_s014
                                 + 1.00117 w_k1_FDX_s014
                                 + 0.741404 w_k1_IBM_s014
                                 + 0.940253 w_k1_INTC_s014
                                 + 0.687689 w_k1_KO_s014
                                 + 0.616245 w_k1_MAT_s014
                                 + 0.790648 w_k1_MSFT_s014
                                 + 0.885187 w_k1_PG_s014 >= 1
 z_k1_s0144_constraint:          var_1 + z_k1_s0144 + 0.588202 w_k1_ADM_s014
                                 + 0.62 w_k1_AXP_s014 + 0.543549 w_k1_DIS_s014
                                 + 0.713956 w_k1_FDX_s014
                                 + 0.775351 w_k1_IBM_s014
                                 + 0.713423 w_k1_INTC_s014
                                 + 0.58935 w_k1_KO_s014
                                 + 0.561827 w_k1_MAT_s014
                                 + 0.753727 w_k1_MSFT_s014
                                 + 0.793962 w_k1_PG_s014 >= 1
 z_k1_s0145_constraint:          var_1 + z_k1_s0145 + 1.19914 w_k1_ADM_s014
                                 + 1.23823 w_k1_AXP_s014 + 1.1491 w_k1_DIS_s014
                                 + 1.2339 w_k1_FDX_s014
                                 + 0.908023 w_k1_IBM_s014
                                 + 1.01762 w_k1_INTC_s014
                                 + 0.873354 w_k1_KO_s014
                                 + 0.826398 w_k1_MAT_s014
                                 + 0.949494 w_k1_MSFT_s014
                                 + 0.967184 w_k1_PG_s014 >= 1
 z_k1_s0146_constraint:          var_1 + z_k1_s0146 + 0.850003 w_k1_ADM_s014
                                 + 0.619205 w_k1_AXP_s014
                                 + 1.04954 w_k1_DIS_s014
                                 + 0.824178 w_k1_FDX_s014
                                 + 0.878351 w_k1_IBM_s014
                                 + 0.855079 w_k1_INTC_s014
                                 + 0.800654 w_k1_KO_s014
                                 + 0.454668 w_k1_MAT_s014
                                 + 0.875146 w_k1_MSFT_s014
                                 + 0.837374 w_k1_PG_s014 >= 1
 z_k1_s0147_constraint:          var_1 + z_k1_s0147 + 0.995034 w_k1_ADM_s014
                                 + 0.810993 w_k1_AXP_s014
                                 + 0.93945 w_k1_DIS_s014
                                 + 0.92207 w_k1_FDX_s014
                                 + 1.09023 w_k1_IBM_s014
                                 + 1.03911 w_k1_INTC_s014
                                 + 0.971535 w_k1_KO_s014
                                 + 0.594128 w_k1_MAT_s014
                                 + 0.975839 w_k1_MSFT_s014
                                 + 0.935427 w_k1_PG_s014 >= 1
 z_k1_s0150_constraint:          var_1 + z_k1_s0150 + 0.809322 w_k1_ADM_s015
                                 + 0.602603 w_k1_AXP_s015
                                 + 0.683104 w_k1_DIS_s015
                                 + 0.835079 w_k1_FDX_s015
                                 + 0.661673 w_k1_IBM_s015
                                 + 1.01401 w_k1_INTC_s015
                                 + 0.70825 w_k1_KO_s015
                                 + 0.686301 w_k1_MAT_s015
                                 + 0.972371 w_k1_MSFT_s015
                                 + 0.877897 w_k1_PG_s015 >= 1
 z_k1_s0151_constraint:          var_1 + z_k1_s0151 + 1.10162 w_k1_ADM_s015
                                 + 1.09853 w_k1_AXP_s015
                                 + 0.907538 w_k1_DIS_s015
                                 + 0.94225 w_k1_FDX_s015
                                 + 1.28328 w_k1_IBM_s015
                                 + 1.18436 w_k1_INTC_s015
                                 + 1.20461 w_k1_KO_s015 + 1.60376 w_k1_MAT_s015
                                 + 0.975231 w_k1_MSFT_s015
                                 + 1.01972 w_k1_PG_s015 >= 1
 z_k1_s0152_constraint:          var_1 + z_k1_s0152 + 0.822808 w_k1_ADM_s015
                                 + 0.712397 w_k1_AXP_s015
                                 + 0.692977 w_k1_DIS_s015
                                 + 0.791389 w_k1_FDX_s015
                                 + 0.869778 w_k1_IBM_s015
                                 + 1.16057 w_k1_INTC_s015
                                 + 0.862358 w_k1_KO_s015
                                 + 0.712867 w_k1_MAT_s015
                                 + 0.889805 w_k1_MSFT_s015
                                 + 0.887435 w_k1_PG_s015 >= 1
 z_k1_s0153_constraint:          var_1 + z_k1_s0153 + 1.09402 w_k1_ADM_s015
                                 + 1.13303 w_k1_AXP_s015
                                 + 0.850842 w_k1_DIS_s015
                                 + 0.914095 w_k1_FDX_s015
                                 + 0.817586 w_k1_IBM_s015
                                 + 1.16937 w_k1_INTC_s015
                                 + 0.92654 w_k1_KO_s015
                                 + 0.795422 w_k1_MAT_s015
                                 + 0.909187 w_k1_MSFT_s015
                                 + 1.00394 w_k1_PG_s015 >= 1
 z_k1_s0154_constraint:          var_1 + z_k1_s0154 + 1.03944 w_k1_ADM_s015
                                 + 1.02759 w_k1_AXP_s015
                                 + 0.957108 w_k1_DIS_s015
                                 + 1.09548 w_k1_FDX_s015
                                 + 0.849766 w_k1_IBM_s015
                                 + 1.2639 w_k1_INTC_s015
                                 + 0.882188 w_k1_KO_s015
                                 + 1.05031 w_k1_MAT_s015
                                 + 1.01865 w_k1_MSFT_s015
                                 + 1.00754 w_k1_PG_s015 >= 1
 z_k1_s0155_constraint:          var_1 + z_k1_s0155 + 0.869002 w_k1_ADM_s015
                                 + 0.862781 w_k1_AXP_s015
                                 + 0.630737 w_k1_DIS_s015
                                 + 0.775859 w_k1_FDX_s015
                                 + 1.04343 w_k1_IBM_s015
                                 + 0.887766 w_k1_INTC_s015
                                 + 0.969238 w_k1_KO_s015
                                 + 1.45031 w_k1_MAT_s015
                                 + 0.939847 w_k1_MSFT_s015
                                 + 0.966795 w_k1_PG_s015 >= 1
 z_k1_s0156_constraint:          var_1 + z_k1_s0156 + 1.55909 w_k1_ADM_s015
                                 + 1.25057 w_k1_AXP_s015
                                 + 1.41262 w_k1_DIS_s015
                                 + 1.35598 w_k1_FDX_s015
                                 + 1.12057 w_k1_IBM_s015
                                 + 1.35265 w_k1_INTC_s015 + 1.2945 w_k1_KO_s015
                                 + 0.939261 w_k1_MAT_s015
                                 + 1.29524 w_k1_MSFT_s015
                                 + 1.17952 w_k1_PG_s015 >= 1
 z_k1_s0157_constraint:          var_1 + z_k1_s0157 + 1.1685 w_k1_ADM_s015
                                 + 1.41632 w_k1_AXP_s015
                                 + 1.21758 w_k1_DIS_s015
                                 + 1.08681 w_k1_FDX_s015
                                 + 1.26299 w_k1_IBM_s015
                                 + 1.38856 w_k1_INTC_s015
                                 + 1.31279 w_k1_KO_s015 + 1.63225 w_k1_MAT_s015
                                 + 1.15805 w_k1_MSFT_s015
                                 + 1.18818 w_k1_PG_s015 >= 1
 z_k1_s0160_constraint:          var_1 + z_k1_s0160 + 1.08351 w_k1_ADM_s016
                                 + 1.38597 w_k1_AXP_s016
                                 + 1.59746 w_k1_DIS_s016
                                 + 1.14814 w_k1_FDX_s016
                                 + 1.83667 w_k1_IBM_s016
                                 + 1.34068 w_k1_INTC_s016
                                 + 1.41809 w_k1_KO_s016 + 1.75297 w_k1_MAT_s016
                                 + 1.12892 w_k1_MSFT_s016
                                 + 1.13697 w_k1_PG_s016 >= 1
 z_k1_s0161_constraint:          var_1 + z_k1_s0161 + 1.24607 w_k1_ADM_s016
                                 + 0.931069 w_k1_AXP_s016
                                 + 1.26641 w_k1_DIS_s016
                                 + 1.14448 w_k1_FDX_s016
                                 + 1.05185 w_k1_IBM_s016
                                 + 1.15195 w_k1_INTC_s016
                                 + 1.20032 w_k1_KO_s016
                                 + 0.968833 w_k1_MAT_s016
                                 + 1.22304 w_k1_MSFT_s016
                                 + 1.10035 w_k1_PG_s016 >= 1
 z_k1_s0162_constraint:          var_1 + z_k1_s0162 + 1.07891 w_k1_ADM_s016
                                 + 1.05515 w_k1_AXP_s016
                                 + 1.18757 w_k1_DIS_s016
                                 + 1.22355 w_k1_FDX_s016
                                 + 0.79001 w_k1_IBM_s016
                                 + 1.12758 w_k1_INTC_s016
                                 + 1.01269 w_k1_KO_s016
                                 + 0.851577 w_k1_MAT_s016
                                 + 1.12559 w_k1_MSFT_s016
                                 + 0.930256 w_k1_PG_s016 >= 1
 z_k1_s0163_constraint:          var_1 + z_k1_s0163 + 0.878707 w_k1_ADM_s016
                                 + 0.640771 w_k1_AXP_s016
                                 + 0.754153 w_k1_DIS_s016
                                 + 0.823085 w_k1_FDX_s016
                                 + 0.762499 w_k1_IBM_s016
                                 + 0.915691 w_k1_INTC_s016
                                 + 0.766294 w_k1_KO_s016
                                 + 0.670092 w_k1_MAT_s016
                                 + 0.824715 w_k1_MSFT_s016
                                 + 0.813251 w_k1_PG_s016 >= 1
 z_k1_s0164_constraint:          var_1 + z_k1_s0164 + 0.87147 w_k1_ADM_s016
                                 + 0.845665 w_k1_AXP_s016
                                 + 1.17936 w_k1_DIS_s016 + 0.854 w_k1_FDX_s016
                                 + 1.39921 w_k1_IBM_s016
                                 + 1.03722 w_k1_INTC_s016
                                 + 1.27978 w_k1_KO_s016 + 1.11485 w_k1_MAT_s016
                                 + 0.965008 w_k1_MSFT_s016
                                 + 1.07518 w_k1_PG_s016 >= 1
 z_k1_s0165_constraint:          var_1 + z_k1_s0165 + 1.06878 w_k1_ADM_s016
                                 + 0.961265 w_k1_AXP_s016
                                 + 0.754611 w_k1_DIS_s016
                                 + 0.884097 w_k1_FDX_s016
                                 + 1.17584 w_k1_IBM_s016
                                 + 0.903133 w_k1_INTC_s016
                                 + 1.04551 w_k1_KO_s016 + 1.51704 w_k1_MAT_s016
                                 + 0.807456 w_k1_MSFT_s016
                                 + 1.01724 w_k1_PG_s016 >= 1
 z_k1_s0166_constraint:          var_1 + z_k1_s0166 + 1.092 w_k1_ADM_s016
                                 + 1.36402 w_k1_AXP_s016
                                 + 1.25673 w_k1_DIS_s016
                                 + 1.18095 w_k1_FDX_s016
                                 + 1.28881 w_k1_IBM_s016
                                 + 1.03733 w_k1_INTC_s016
                                 + 1.11364 w_k1_KO_s016 + 1.54333 w_k1_MAT_s016
                                 + 1.10176 w_k1_MSFT_s016
                                 + 1.02391 w_k1_PG_s016 >= 1
 z_k1_s0167_constraint:          var_1 + z_k1_s0167 + 0.848224 w_k1_ADM_s016
                                 + 0.968155 w_k1_AXP_s016
                                 + 0.693449 w_k1_DIS_s016
                                 + 0.899794 w_k1_FDX_s016
                                 + 0.852668 w_k1_IBM_s016
                                 + 1.08237 w_k1_INTC_s016
                                 + 0.838515 w_k1_KO_s016
                                 + 0.990845 w_k1_MAT_s016
                                 + 0.886815 w_k1_MSFT_s016
                                 + 0.85679 w_k1_PG_s016 >= 1
 z_k1_s0170_constraint:          var_1 + z_k1_s0170 + 0.905461 w_k1_ADM_s017
                                 + 0.897604 w_k1_AXP_s017
                                 + 0.581653 w_k1_DIS_s017
                                 + 0.756315 w_k1_FDX_s017
                                 + 0.835687 w_k1_IBM_s017
                                 + 0.775379 w_k1_INTC_s017
                                 + 0.739218 w_k1_KO_s017
                                 + 0.882287 w_k1_MAT_s017
                                 + 0.688496 w_k1_MSFT_s017
                                 + 0.763062 w_k1_PG_s017 >= 1
 z_k1_s0171_constraint:          var_1 + z_k1_s0171 + 0.82274 w_k1_ADM_s017
                                 + 1.01245 w_k1_AXP_s017
                                 + 1.13734 w_k1_DIS_s017
                                 + 1.02848 w_k1_FDX_s017
                                 + 0.879258 w_k1_IBM_s017
                                 + 1.21719 w_k1_INTC_s017
                                 + 1.01827 w_k1_KO_s017 + 1.118 w_k1_MAT_s017
                                 + 1.09075 w_k1_MSFT_s017 + 0.9861 w_k1_PG_s017
                                 >= 1
 z_k1_s0172_constraint:          var_1 + z_k1_s0172 + 1.28103 w_k1_ADM_s017
                                 + 1.71377 w_k1_AXP_s017
                                 + 1.24739 w_k1_DIS_s017
                                 + 1.34271 w_k1_FDX_s017
                                 + 0.91926 w_k1_IBM_s017
                                 + 1.48296 w_k1_INTC_s017
                                 + 1.04999 w_k1_KO_s017 + 1.24561 w_k1_MAT_s017
                                 + 1.19838 w_k1_MSFT_s017
                                 + 1.11539 w_k1_PG_s017 >= 1
 z_k1_s0173_constraint:          var_1 + z_k1_s0173 + 0.993355 w_k1_ADM_s017
                                 + 1.23436 w_k1_AXP_s017
                                 + 0.788584 w_k1_DIS_s017
                                 + 0.964609 w_k1_FDX_s017
                                 + 1.01614 w_k1_IBM_s017
                                 + 1.01066 w_k1_INTC_s017
                                 + 0.823901 w_k1_KO_s017
                                 + 0.983239 w_k1_MAT_s017
                                 + 0.857139 w_k1_MSFT_s017
                                 + 0.914185 w_k1_PG_s017 >= 1
 z_k1_s0174_constraint:          var_1 + z_k1_s0174 + 0.881486 w_k1_ADM_s017
                                 + 0.860667 w_k1_AXP_s017
                                 + 0.825842 w_k1_DIS_s017
                                 + 0.983695 w_k1_FDX_s017
                                 + 0.609487 w_k1_IBM_s017
                                 + 0.918284 w_k1_INTC_s017
                                 + 0.768319 w_k1_KO_s017
                                 + 0.588459 w_k1_MAT_s017
                                 + 0.960958 w_k1_MSFT_s017
                                 + 0.707542 w_k1_PG_s017 >= 1
 z_k1_s0175_constraint:          var_1 + z_k1_s0175 + 0.982709 w_k1_ADM_s017
                                 + 0.831679 w_k1_AXP_s017
                                 + 0.773901 w_k1_DIS_s017
                                 + 0.824466 w_k1_FDX_s017
                                 + 1.38417 w_k1_IBM_s017
                                 + 0.821278 w_k1_INTC_s017
                                 + 0.919986 w_k1_KO_s017
                                 + 1.75769 w_k1_MAT_s017
                                 + 0.858513 w_k1_MSFT_s017
                                 + 0.906419 w_k1_PG_s017 >= 1
 z_k1_s0176_constraint:          var_1 + z_k1_s0176 + 0.727058 w_k1_ADM_s017
                                 + 1.33104 w_k1_AXP_s017
                                 + 1.46549 w_k1_DIS_s017
                                 + 1.18922 w_k1_FDX_s017
                                 + 1.56364 w_k1_IBM_s017
                                 + 1.09096 w_k1_INTC_s017
                                 + 1.37113 w_k1_KO_s017 + 1.82207 w_k1_MAT_s017
                                 + 1.17829 w_k1_MSFT_s017
                                 + 1.14515 w_k1_PG_s017 >= 1
 z_k1_s0177_constraint:          var_1 + z_k1_s0177 + 0.930888 w_k1_ADM_s017
                                 + 0.773199 w_k1_AXP_s017
                                 + 0.833678 w_k1_DIS_s017
                                 + 0.894384 w_k1_FDX_s017
                                 + 0.86871 w_k1_IBM_s017
                                 + 0.954737 w_k1_INTC_s017
                                 + 0.814631 w_k1_KO_s017
                                 + 0.58535 w_k1_MAT_s017
                                 + 0.998006 w_k1_MSFT_s017
                                 + 0.797663 w_k1_PG_s017 >= 1
 z_k1_s1000_constraint:          var_1 + z_k1_s1000 + 0.913988 w_k1_ADM_s100
                                 + 0.973811 w_k1_AXP_s100
                                 + 1.03171 w_k1_DIS_s100
                                 + 1.04509 w_k1_FDX_s100
                                 + 0.921107 w_k1_IBM_s100
                                 + 0.977471 w_k1_INTC_s100
                                 + 1.10943 w_k1_KO_s100
                                 + 0.878903 w_k1_MAT_s100
                                 + 1.08651 w_k1_MSFT_s100
                                 + 1.07273 w_k1_PG_s100 >= 1
 z_k1_s1001_constraint:          var_1 + z_k1_s1001 + 0.654303 w_k1_ADM_s100
                                 + 0.822434 w_k1_AXP_s100
                                 + 0.676426 w_k1_DIS_s100
                                 + 0.801512 w_k1_FDX_s100
                                 + 1.05452 w_k1_IBM_s100
                                 + 0.912317 w_k1_INTC_s100
                                 + 0.987116 w_k1_KO_s100 + 1.086 w_k1_MAT_s100
                                 + 0.955896 w_k1_MSFT_s100
                                 + 0.926503 w_k1_PG_s100 >= 1
 z_k1_s1002_constraint:          var_1 + z_k1_s1002 + 1.06944 w_k1_ADM_s100
                                 + 0.940677 w_k1_AXP_s100
                                 + 1.0253 w_k1_DIS_s100
                                 + 0.907311 w_k1_FDX_s100
                                 + 1.35365 w_k1_IBM_s100
                                 + 0.946574 w_k1_INTC_s100
                                 + 1.23916 w_k1_KO_s100 + 1.48141 w_k1_MAT_s100
                                 + 1.06363 w_k1_MSFT_s100
                                 + 1.06002 w_k1_PG_s100 >= 1
 z_k1_s1003_constraint:          var_1 + z_k1_s1003 + 0.752603 w_k1_ADM_s100
                                 + 0.613729 w_k1_AXP_s100
                                 + 0.727503 w_k1_DIS_s100
                                 + 0.75253 w_k1_FDX_s100
                                 + 0.617841 w_k1_IBM_s100
                                 + 0.786546 w_k1_INTC_s100
                                 + 0.820985 w_k1_KO_s100
                                 + 0.601608 w_k1_MAT_s100
                                 + 0.827426 w_k1_MSFT_s100
                                 + 0.757748 w_k1_PG_s100 >= 1
 z_k1_s1004_constraint:          var_1 + z_k1_s1004 + 1.1407 w_k1_ADM_s100
                                 + 1.36424 w_k1_AXP_s100
                                 + 1.40452 w_k1_DIS_s100
                                 + 1.08795 w_k1_FDX_s100
                                 + 1.59125 w_k1_IBM_s100
                                 + 1.1315 w_k1_INTC_s100 + 1.58292 w_k1_KO_s100
                                 + 1.87664 w_k1_MAT_s100
                                 + 1.20921 w_k1_MSFT_s100
                                 + 1.22304 w_k1_PG_s100 >= 1
 z_k1_s1005_constraint:          var_1 + z_k1_s1005 + 1.01171 w_k1_ADM_s100
                                 + 1.14347 w_k1_AXP_s100
                                 + 1.46046 w_k1_DIS_s100
                                 + 1.27187 w_k1_FDX_s100
                                 + 0.969111 w_k1_IBM_s100
                                 + 1.1872 w_k1_INTC_s100 + 1.21909 w_k1_KO_s100
                                 + 0.724201 w_k1_MAT_s100
                                 + 1.3287 w_k1_MSFT_s100 + 1.21974 w_k1_PG_s100
                                 >= 1
 z_k1_s1006_constraint:          var_1 + z_k1_s1006 + 1.17773 w_k1_ADM_s100
                                 + 1.57675 w_k1_AXP_s100
                                 + 1.32302 w_k1_DIS_s100 + 1.4657 w_k1_FDX_s100
                                 + 1.2514 w_k1_IBM_s100
                                 + 1.15019 w_k1_INTC_s100
                                 + 1.34027 w_k1_KO_s100 + 1.6597 w_k1_MAT_s100
                                 + 1.20795 w_k1_MSFT_s100
                                 + 1.18653 w_k1_PG_s100 >= 1
 z_k1_s1007_constraint:          var_1 + z_k1_s1007 + 0.882588 w_k1_ADM_s100
                                 + 0.902279 w_k1_AXP_s100
                                 + 0.802437 w_k1_DIS_s100
                                 + 0.867281 w_k1_FDX_s100
                                 + 0.748037 w_k1_IBM_s100
                                 + 0.835601 w_k1_INTC_s100
                                 + 0.868389 w_k1_KO_s100
                                 + 0.621526 w_k1_MAT_s100
                                 + 0.887901 w_k1_MSFT_s100
                                 + 0.916075 w_k1_PG_s100 >= 1
 z_k1_s1010_constraint:          var_1 + z_k1_s1010 + 1.4961 w_k1_ADM_s101
                                 + 1.2034 w_k1_AXP_s101 + 1.10457 w_k1_DIS_s101
                                 + 1.00647 w_k1_FDX_s101
                                 + 1.39036 w_k1_IBM_s101
                                 + 1.04888 w_k1_INTC_s101 + 1.1837 w_k1_KO_s101
                                 + 1.10484 w_k1_MAT_s101
                                 + 1.0989 w_k1_MSFT_s101 + 1.13756 w_k1_PG_s101
                                 >= 1
 z_k1_s1011_constraint:          var_1 + z_k1_s1011 + 1.18095 w_k1_ADM_s101
                                 + 0.954753 w_k1_AXP_s101
                                 + 0.866349 w_k1_DIS_s101
                                 + 0.980865 w_k1_FDX_s101
                                 + 0.741688 w_k1_IBM_s101
                                 + 0.929036 w_k1_INTC_s101
                                 + 0.892336 w_k1_KO_s101
                                 + 0.63952 w_k1_MAT_s101
                                 + 0.817569 w_k1_MSFT_s101
                                 + 0.885335 w_k1_PG_s101 >= 1
 z_k1_s1012_constraint:          var_1 + z_k1_s1012 + 1.58609 w_k1_ADM_s101
                                 + 0.993479 w_k1_AXP_s101
                                 + 1.62279 w_k1_DIS_s101
                                 + 1.26906 w_k1_FDX_s101
                                 + 0.962213 w_k1_IBM_s101
                                 + 1.07953 w_k1_INTC_s101
                                 + 1.24747 w_k1_KO_s101
                                 + 0.638285 w_k1_MAT_s101
                                 + 1.13059 w_k1_MSFT_s101
                                 + 1.09906 w_k1_PG_s101 >= 1
 z_k1_s1013_constraint:          var_1 + z_k1_s1013 + 0.92676 w_k1_ADM_s101
                                 + 0.721548 w_k1_AXP_s101
                                 + 1.15542 w_k1_DIS_s101 + 1.0582 w_k1_FDX_s101
                                 + 0.836334 w_k1_IBM_s101
                                 + 1.19305 w_k1_INTC_s101
                                 + 1.09365 w_k1_KO_s101 + 0.6278 w_k1_MAT_s101
                                 + 1.15477 w_k1_MSFT_s101
                                 + 1.02467 w_k1_PG_s101 >= 1
 z_k1_s1014_constraint:          var_1 + z_k1_s1014 + 1.22896 w_k1_ADM_s101
                                 + 0.658584 w_k1_AXP_s101
                                 + 0.799248 w_k1_DIS_s101
                                 + 0.813441 w_k1_FDX_s101
                                 + 0.918739 w_k1_IBM_s101
                                 + 0.875539 w_k1_INTC_s101
                                 + 0.860333 w_k1_KO_s101
                                 + 0.506546 w_k1_MAT_s101
                                 + 0.828537 w_k1_MSFT_s101
                                 + 0.907559 w_k1_PG_s101 >= 1
 z_k1_s1015_constraint:          var_1 + z_k1_s1015 + 1.51517 w_k1_ADM_s101
                                 + 1.15903 w_k1_AXP_s101 + 1.2377 w_k1_DIS_s101
                                 + 1.26557 w_k1_FDX_s101
                                 + 0.863668 w_k1_IBM_s101
                                 + 1.23012 w_k1_INTC_s101 + 1.0335 w_k1_KO_s101
                                 + 0.691958 w_k1_MAT_s101
                                 + 1.32425 w_k1_MSFT_s101
                                 + 1.14254 w_k1_PG_s101 >= 1
 z_k1_s1016_constraint:          var_1 + z_k1_s1016 + 1.30499 w_k1_ADM_s101
                                 + 0.944285 w_k1_AXP_s101
                                 + 1.00038 w_k1_DIS_s101
                                 + 1.19982 w_k1_FDX_s101
                                 + 0.785037 w_k1_IBM_s101
                                 + 1.12304 w_k1_INTC_s101 + 1.0553 w_k1_KO_s101
                                 + 0.701767 w_k1_MAT_s101
                                 + 1.09357 w_k1_MSFT_s101
                                 + 0.983603 w_k1_PG_s101 >= 1
 z_k1_s1017_constraint:          var_1 + z_k1_s1017 + 1.43387 w_k1_ADM_s101
                                 + 1.62459 w_k1_AXP_s101
                                 + 1.62209 w_k1_DIS_s101
                                 + 1.38892 w_k1_FDX_s101
                                 + 1.29336 w_k1_IBM_s101
                                 + 1.26541 w_k1_INTC_s101
                                 + 1.39673 w_k1_KO_s101 + 1.46686 w_k1_MAT_s101
                                 + 1.36323 w_k1_MSFT_s101
                                 + 1.36201 w_k1_PG_s101 >= 1
 z_k1_s1020_constraint:          var_1 + z_k1_s1020 + 1.103 w_k1_ADM_s102
                                 + 1.0773 w_k1_AXP_s102 + 1.47588 w_k1_DIS_s102
                                 + 1.2062 w_k1_FDX_s102 + 1.91498 w_k1_IBM_s102
                                 + 1.4654 w_k1_INTC_s102 + 1.5703 w_k1_KO_s102
                                 + 1.46801 w_k1_MAT_s102
                                 + 1.44195 w_k1_MSFT_s102 + 1.3256 w_k1_PG_s102
                                 >= 1
 z_k1_s1021_constraint:          var_1 + z_k1_s1021 + 0.810946 w_k1_ADM_s102
                                 + 0.708062 w_k1_AXP_s102
                                 + 0.644395 w_k1_DIS_s102
                                 + 0.751412 w_k1_FDX_s102
                                 + 0.676994 w_k1_IBM_s102
                                 + 0.856128 w_k1_INTC_s102
                                 + 0.799805 w_k1_KO_s102
                                 + 0.641532 w_k1_MAT_s102
                                 + 0.873878 w_k1_MSFT_s102
                                 + 0.84716 w_k1_PG_s102 >= 1
 z_k1_s1022_constraint:          var_1 + z_k1_s1022 + 1.10995 w_k1_ADM_s102
                                 + 1.00819 w_k1_AXP_s102
                                 + 0.797567 w_k1_DIS_s102
                                 + 0.977298 w_k1_FDX_s102
                                 + 0.980703 w_k1_IBM_s102
                                 + 1.0875 w_k1_INTC_s102 + 1.04374 w_k1_KO_s102
                                 + 1.31725 w_k1_MAT_s102
                                 + 1.14713 w_k1_MSFT_s102
                                 + 0.981782 w_k1_PG_s102 >= 1
 z_k1_s1023_constraint:          var_1 + z_k1_s1023 + 1.36913 w_k1_ADM_s102
                                 + 0.957275 w_k1_AXP_s102
                                 + 1.03788 w_k1_DIS_s102
                                 + 1.17549 w_k1_FDX_s102
                                 + 0.932707 w_k1_IBM_s102
                                 + 0.883835 w_k1_INTC_s102
                                 + 1.03259 w_k1_KO_s102 + 1.09433 w_k1_MAT_s102
                                 + 1.17935 w_k1_MSFT_s102
                                 + 1.12697 w_k1_PG_s102 >= 1
 z_k1_s1024_constraint:          var_1 + z_k1_s1024 + 1.1191 w_k1_ADM_s102
                                 + 0.711741 w_k1_AXP_s102
                                 + 0.672674 w_k1_DIS_s102
                                 + 0.86193 w_k1_FDX_s102
                                 + 0.997007 w_k1_IBM_s102
                                 + 0.810082 w_k1_INTC_s102
                                 + 0.862688 w_k1_KO_s102
                                 + 1.04725 w_k1_MAT_s102
                                 + 0.818409 w_k1_MSFT_s102
                                 + 0.857226 w_k1_PG_s102 >= 1
 z_k1_s1025_constraint:          var_1 + z_k1_s1025 + 1.05398 w_k1_ADM_s102
                                 + 1.08608 w_k1_AXP_s102
                                 + 0.898436 w_k1_DIS_s102
                                 + 1.08762 w_k1_FDX_s102
                                 + 1.42768 w_k1_IBM_s102
                                 + 1.26403 w_k1_INTC_s102
                                 + 1.16687 w_k1_KO_s102 + 1.58751 w_k1_MAT_s102
                                 + 1.22659 w_k1_MSFT_s102
                                 + 1.13274 w_k1_PG_s102 >= 1
 z_k1_s1026_constraint:          var_1 + z_k1_s1026 + 1.31309 w_k1_ADM_s102
                                 + 1.29333 w_k1_AXP_s102
                                 + 1.35189 w_k1_DIS_s102
                                 + 1.33113 w_k1_FDX_s102
                                 + 1.05728 w_k1_IBM_s102
                                 + 1.21199 w_k1_INTC_s102
                                 + 1.24776 w_k1_KO_s102 + 1.81073 w_k1_MAT_s102
                                 + 1.50902 w_k1_MSFT_s102
                                 + 1.30352 w_k1_PG_s102 >= 1
 z_k1_s1027_constraint:          var_1 + z_k1_s1027 + 0.909233 w_k1_ADM_s102
                                 + 0.917271 w_k1_AXP_s102
                                 + 1.03709 w_k1_DIS_s102
                                 + 1.10672 w_k1_FDX_s102
                                 + 0.688516 w_k1_IBM_s102
                                 + 1.1657 w_k1_INTC_s102 + 0.88698 w_k1_KO_s102
                                 + 0.598054 w_k1_MAT_s102
                                 + 1.21426 w_k1_MSFT_s102
                                 + 0.904021 w_k1_PG_s102 >= 1
 z_k1_s1030_constraint:          var_1 + z_k1_s1030 + 0.781816 w_k1_ADM_s103
                                 + 0.864376 w_k1_AXP_s103
                                 + 0.883891 w_k1_DIS_s103
                                 + 0.817213 w_k1_FDX_s103
                                 + 1.09083 w_k1_IBM_s103
                                 + 0.923873 w_k1_INTC_s103
                                 + 1.22175 w_k1_KO_s103 + 1.43083 w_k1_MAT_s103
                                 + 0.893669 w_k1_MSFT_s103
                                 + 0.852188 w_k1_PG_s103 >= 1
 z_k1_s1031_constraint:          var_1 + z_k1_s1031 + 1.00046 w_k1_ADM_s103
                                 + 1.20517 w_k1_AXP_s103
                                 + 0.970171 w_k1_DIS_s103
                                 + 0.956349 w_k1_FDX_s103
                                 + 0.869351 w_k1_IBM_s103
                                 + 0.902605 w_k1_INTC_s103
                                 + 0.982831 w_k1_KO_s103
                                 + 0.974066 w_k1_MAT_s103
                                 + 1.07144 w_k1_MSFT_s103
                                 + 0.896673 w_k1_PG_s103 >= 1
 z_k1_s1032_constraint:          var_1 + z_k1_s1032 + 1.02078 w_k1_ADM_s103
                                 + 0.910082 w_k1_AXP_s103
                                 + 0.901202 w_k1_DIS_s103
                                 + 0.919608 w_k1_FDX_s103
                                 + 0.645725 w_k1_IBM_s103
                                 + 0.868356 w_k1_INTC_s103
                                 + 0.914683 w_k1_KO_s103
                                 + 0.750525 w_k1_MAT_s103
                                 + 1.01821 w_k1_MSFT_s103
                                 + 0.948551 w_k1_PG_s103 >= 1
 z_k1_s1033_constraint:          var_1 + z_k1_s1033 + 0.962322 w_k1_ADM_s103
                                 + 1.37723 w_k1_AXP_s103
                                 + 1.16842 w_k1_DIS_s103
                                 + 1.02503 w_k1_FDX_s103
                                 + 1.18065 w_k1_IBM_s103
                                 + 1.21498 w_k1_INTC_s103
                                 + 1.28726 w_k1_KO_s103 + 1.39086 w_k1_MAT_s103
                                 + 0.979398 w_k1_MSFT_s103
                                 + 0.932358 w_k1_PG_s103 >= 1
 z_k1_s1034_constraint:          var_1 + z_k1_s1034 + 0.733768 w_k1_ADM_s103
                                 + 0.829457 w_k1_AXP_s103
                                 + 0.706462 w_k1_DIS_s103
                                 + 0.854212 w_k1_FDX_s103
                                 + 0.748772 w_k1_IBM_s103
                                 + 0.966694 w_k1_INTC_s103
                                 + 0.885979 w_k1_KO_s103
                                 + 0.86435 w_k1_MAT_s103
                                 + 0.878655 w_k1_MSFT_s103
                                 + 0.782493 w_k1_PG_s103 >= 1
 z_k1_s1035_constraint:          var_1 + z_k1_s1035 + 0.906768 w_k1_ADM_s103
                                 + 0.756182 w_k1_AXP_s103
                                 + 0.553758 w_k1_DIS_s103
                                 + 0.704516 w_k1_FDX_s103
                                 + 0.927892 w_k1_IBM_s103
                                 + 0.784662 w_k1_INTC_s103
                                 + 0.811734 w_k1_KO_s103
                                 + 0.856452 w_k1_MAT_s103
                                 + 0.825801 w_k1_MSFT_s103
                                 + 0.799625 w_k1_PG_s103 >= 1
 z_k1_s1036_constraint:          var_1 + z_k1_s1036 + 1.07012 w_k1_ADM_s103
                                 + 1.39892 w_k1_AXP_s103
                                 + 1.21555 w_k1_DIS_s103
                                 + 1.16865 w_k1_FDX_s103
                                 + 1.01153 w_k1_IBM_s103
                                 + 1.14972 w_k1_INTC_s103
                                 + 1.08969 w_k1_KO_s103 + 1.03164 w_k1_MAT_s103
                                 + 1.25647 w_k1_MSFT_s103
                                 + 1.06083 w_k1_PG_s103 >= 1
 z_k1_s1037_constraint:          var_1 + z_k1_s1037 + 1.35083 w_k1_ADM_s103
                                 + 1.41166 w_k1_AXP_s103
                                 + 1.61252 w_k1_DIS_s103
                                 + 1.28711 w_k1_FDX_s103 + 1.4104 w_k1_IBM_s103
                                 + 1.20023 w_k1_INTC_s103
                                 + 1.41782 w_k1_KO_s103 + 1.62044 w_k1_MAT_s103
                                 + 1.26451 w_k1_MSFT_s103
                                 + 1.26643 w_k1_PG_s103 >= 1
 z_k1_s1040_constraint:          var_1 + z_k1_s1040 + 1.51738 w_k1_ADM_s104
                                 + 1.05685 w_k1_AXP_s104
                                 + 1.49414 w_k1_DIS_s104
                                 + 1.42169 w_k1_FDX_s104
                                 + 0.984264 w_k1_IBM_s104
                                 + 1.5119 w_k1_INTC_s104 + 1.4165 w_k1_KO_s104
                                 + 0.556536 w_k1_MAT_s104
                                 + 1.33769 w_k1_MSFT_s104
                                 + 1.33332 w_k1_PG_s104 >= 1
 z_k1_s1041_constraint:          var_1 + z_k1_s1041 + 1.30905 w_k1_ADM_s104
                                 + 0.912163 w_k1_AXP_s104
                                 + 1.15924 w_k1_DIS_s104
                                 + 1.03212 w_k1_FDX_s104
                                 + 1.38386 w_k1_IBM_s104
                                 + 0.977144 w_k1_INTC_s104
                                 + 1.34472 w_k1_KO_s104 + 1.08587 w_k1_MAT_s104
                                 + 0.996858 w_k1_MSFT_s104
                                 + 1.12959 w_k1_PG_s104 >= 1
 z_k1_s1042_constraint:          var_1 + z_k1_s1042 + 1.48226 w_k1_ADM_s104
                                 + 1.45287 w_k1_AXP_s104
                                 + 1.18777 w_k1_DIS_s104
                                 + 1.29781 w_k1_FDX_s104
                                 + 1.12858 w_k1_IBM_s104
                                 + 1.1875 w_k1_INTC_s104 + 1.16349 w_k1_KO_s104
                                 + 0.972719 w_k1_MAT_s104
                                 + 1.05706 w_k1_MSFT_s104
                                 + 0.958118 w_k1_PG_s104 >= 1
 z_k1_s1043_constraint:          var_1 + z_k1_s1043 + 1.22588 w_k1_ADM_s104
                                 + 0.670492 w_k1_AXP_s104
                                 + 1.26296 w_k1_DIS_s104
                                 + 1.07608 w_k1_FDX_s104
                                 + 0.906566 w_k1_IBM_s104
                                 + 0.968938 w_k1_INTC_s104
                                 + 1.11437 w_k1_KO_s104
                                 + 0.482423 w_k1_MAT_s104
                                 + 0.962309 w_k1_MSFT_s104
                                 + 1.00288 w_k1_PG_s104 >= 1
 z_k1_s1044_constraint:          var_1 + z_k1_s1044 + 0.95453 w_k1_ADM_s104
                                 + 0.829513 w_k1_AXP_s104
                                 + 0.707215 w_k1_DIS_s104
                                 + 0.937534 w_k1_FDX_s104
                                 + 0.660445 w_k1_IBM_s104
                                 + 0.970806 w_k1_INTC_s104
                                 + 0.775638 w_k1_KO_s104
                                 + 0.680485 w_k1_MAT_s104
                                 + 0.869231 w_k1_MSFT_s104
                                 + 0.848547 w_k1_PG_s104 >= 1
 z_k1_s1045_constraint:          var_1 + z_k1_s1045 + 1.17824 w_k1_ADM_s104
                                 + 0.923214 w_k1_AXP_s104
                                 + 1.01448 w_k1_DIS_s104
                                 + 1.16756 w_k1_FDX_s104
                                 + 0.765674 w_k1_IBM_s104
                                 + 1.12846 w_k1_INTC_s104
                                 + 0.86392 w_k1_KO_s104
                                 + 0.468955 w_k1_MAT_s104
                                 + 0.961924 w_k1_MSFT_s104
                                 + 1.08819 w_k1_PG_s104 >= 1
 z_k1_s1046_constraint:          var_1 + z_k1_s1046 + 0.79357 w_k1_ADM_s104
                                 + 0.71878 w_k1_AXP_s104
                                 + 0.69416 w_k1_DIS_s104
                                 + 0.818989 w_k1_FDX_s104
                                 + 0.996778 w_k1_IBM_s104
                                 + 0.860319 w_k1_INTC_s104
                                 + 0.941841 w_k1_KO_s104
                                 + 0.846207 w_k1_MAT_s104
                                 + 0.824622 w_k1_MSFT_s104
                                 + 0.813308 w_k1_PG_s104 >= 1
 z_k1_s1047_constraint:          var_1 + z_k1_s1047 + 1.21679 w_k1_ADM_s104
                                 + 1.45285 w_k1_AXP_s104
                                 + 1.72224 w_k1_DIS_s104 + 1.2651 w_k1_FDX_s104
                                 + 2.28872 w_k1_IBM_s104
                                 + 1.46217 w_k1_INTC_s104
                                 + 1.79551 w_k1_KO_s104 + 1.85744 w_k1_MAT_s104
                                 + 1.23397 w_k1_MSFT_s104
                                 + 1.40042 w_k1_PG_s104 >= 1
 z_k1_s1050_constraint:          var_1 + z_k1_s1050 + 1.28836 w_k1_ADM_s105
                                 + 1.40873 w_k1_AXP_s105
                                 + 1.28999 w_k1_DIS_s105
                                 + 1.10029 w_k1_FDX_s105
                                 + 1.08702 w_k1_IBM_s105
                                 + 1.14162 w_k1_INTC_s105
                                 + 1.11233 w_k1_KO_s105 + 1.37848 w_k1_MAT_s105
                                 + 1.00268 w_k1_MSFT_s105
                                 + 1.01938 w_k1_PG_s105 >= 1
 z_k1_s1051_constraint:          var_1 + z_k1_s1051 + 0.919992 w_k1_ADM_s105
                                 + 1.53789 w_k1_AXP_s105 + 1.3813 w_k1_DIS_s105
                                 + 1.15677 w_k1_FDX_s105 + 1.8691 w_k1_IBM_s105
                                 + 1.15288 w_k1_INTC_s105
                                 + 1.71826 w_k1_KO_s105 + 2.20594 w_k1_MAT_s105
                                 + 1.04106 w_k1_MSFT_s105
                                 + 1.30051 w_k1_PG_s105 >= 1
 z_k1_s1052_constraint:          var_1 + z_k1_s1052 + 1.27854 w_k1_ADM_s105
                                 + 1.85897 w_k1_AXP_s105
                                 + 1.64994 w_k1_DIS_s105
                                 + 1.37343 w_k1_FDX_s105 + 1.3158 w_k1_IBM_s105
                                 + 1.11972 w_k1_INTC_s105
                                 + 1.24963 w_k1_KO_s105 + 1.49268 w_k1_MAT_s105
                                 + 1.21958 w_k1_MSFT_s105
                                 + 1.14073 w_k1_PG_s105 >= 1
 z_k1_s1053_constraint:          var_1 + z_k1_s1053 + 1.17784 w_k1_ADM_s105
                                 + 0.939548 w_k1_AXP_s105
                                 + 1.09721 w_k1_DIS_s105
                                 + 1.03077 w_k1_FDX_s105
                                 + 0.914883 w_k1_IBM_s105
                                 + 1.01055 w_k1_INTC_s105
                                 + 1.13227 w_k1_KO_s105 + 0.92686 w_k1_MAT_s105
                                 + 1.18767 w_k1_MSFT_s105
                                 + 1.04625 w_k1_PG_s105 >= 1
 z_k1_s1054_constraint:          var_1 + z_k1_s1054 + 0.857984 w_k1_ADM_s105
                                 + 0.852721 w_k1_AXP_s105
                                 + 0.745044 w_k1_DIS_s105
                                 + 0.844087 w_k1_FDX_s105
                                 + 1.13502 w_k1_IBM_s105
                                 + 0.776955 w_k1_INTC_s105
                                 + 0.912651 w_k1_KO_s105
                                 + 1.17234 w_k1_MAT_s105
                                 + 0.713843 w_k1_MSFT_s105
                                 + 0.833082 w_k1_PG_s105 >= 1
 z_k1_s1055_constraint:          var_1 + z_k1_s1055 + 1.01481 w_k1_ADM_s105
                                 + 0.979464 w_k1_AXP_s105
                                 + 0.986327 w_k1_DIS_s105
                                 + 0.935554 w_k1_FDX_s105
                                 + 1.37079 w_k1_IBM_s105
                                 + 0.805302 w_k1_INTC_s105
                                 + 0.993659 w_k1_KO_s105 + 1.2575 w_k1_MAT_s105
                                 + 0.893615 w_k1_MSFT_s105
                                 + 0.943993 w_k1_PG_s105 >= 1
 z_k1_s1056_constraint:          var_1 + z_k1_s1056 + 1.10994 w_k1_ADM_s105
                                 + 1.14326 w_k1_AXP_s105
                                 + 0.848808 w_k1_DIS_s105
                                 + 1.07793 w_k1_FDX_s105
                                 + 0.67619 w_k1_IBM_s105
                                 + 0.967067 w_k1_INTC_s105
                                 + 0.954715 w_k1_KO_s105
                                 + 0.795349 w_k1_MAT_s105
                                 + 0.944747 w_k1_MSFT_s105
                                 + 0.948558 w_k1_PG_s105 >= 1
 z_k1_s1057_constraint:          var_1 + z_k1_s1057 + 1.01014 w_k1_ADM_s105
                                 + 0.781464 w_k1_AXP_s105
                                 + 0.758456 w_k1_DIS_s105
                                 + 0.822814 w_k1_FDX_s105
                                 + 0.647137 w_k1_IBM_s105
                                 + 0.734324 w_k1_INTC_s105
                                 + 0.795354 w_k1_KO_s105
                                 + 0.706967 w_k1_MAT_s105
                                 + 0.728092 w_k1_MSFT_s105
                                 + 0.85336 w_k1_PG_s105 >= 1
 z_k1_s1060_constraint:          var_1 + z_k1_s1060 + 1.1773 w_k1_ADM_s106
                                 + 1.62811 w_k1_AXP_s106
                                 + 1.63263 w_k1_DIS_s106
                                 + 1.41591 w_k1_FDX_s106
                                 + 1.39251 w_k1_IBM_s106
                                 + 1.01957 w_k1_INTC_s106
                                 + 1.47695 w_k1_KO_s106 + 2.28138 w_k1_MAT_s106
                                 + 1.24145 w_k1_MSFT_s106
                                 + 1.21405 w_k1_PG_s106 >= 1
 z_k1_s1061_constraint:          var_1 + z_k1_s1061 + 1.01553 w_k1_ADM_s106
                                 + 1.13666 w_k1_AXP_s106
                                 + 0.986668 w_k1_DIS_s106
                                 + 1.16476 w_k1_FDX_s106
                                 + 0.724858 w_k1_IBM_s106
                                 + 1.13518 w_k1_INTC_s106
                                 + 0.920653 w_k1_KO_s106
                                 + 0.773201 w_k1_MAT_s106
                                 + 0.969949 w_k1_MSFT_s106
                                 + 0.961324 w_k1_PG_s106 >= 1
 z_k1_s1062_constraint:          var_1 + z_k1_s1062 + 0.964228 w_k1_ADM_s106
                                 + 0.72506 w_k1_AXP_s106 + 1.2247 w_k1_DIS_s106
                                 + 1.02373 w_k1_FDX_s106
                                 + 0.881258 w_k1_IBM_s106
                                 + 1.19049 w_k1_INTC_s106
                                 + 1.08701 w_k1_KO_s106 + 0.46436 w_k1_MAT_s106
                                 + 1.22476 w_k1_MSFT_s106
                                 + 0.979353 w_k1_PG_s106 >= 1
 z_k1_s1063_constraint:          var_1 + z_k1_s1063 + 0.722646 w_k1_ADM_s106
                                 + 1.00366 w_k1_AXP_s106
                                 + 0.936148 w_k1_DIS_s106
                                 + 0.854037 w_k1_FDX_s106
                                 + 1.38736 w_k1_IBM_s106
                                 + 1.13689 w_k1_INTC_s106
                                 + 1.25945 w_k1_KO_s106 + 1.18011 w_k1_MAT_s106
                                 + 0.984408 w_k1_MSFT_s106
                                 + 1.04839 w_k1_PG_s106 >= 1
 z_k1_s1064_constraint:          var_1 + z_k1_s1064 + 0.990647 w_k1_ADM_s106
                                 + 0.932237 w_k1_AXP_s106
                                 + 0.866133 w_k1_DIS_s106
                                 + 0.905764 w_k1_FDX_s106
                                 + 1.01725 w_k1_IBM_s106
                                 + 1.01434 w_k1_INTC_s106
                                 + 1.01766 w_k1_KO_s106
                                 + 0.828007 w_k1_MAT_s106
                                 + 0.943965 w_k1_MSFT_s106
                                 + 1.0684 w_k1_PG_s106 >= 1
 z_k1_s1065_constraint:          var_1 + z_k1_s1065 + 0.631273 w_k1_ADM_s106
                                 + 0.759491 w_k1_AXP_s106
                                 + 0.775887 w_k1_DIS_s106
                                 + 0.812828 w_k1_FDX_s106
                                 + 1.07435 w_k1_IBM_s106
                                 + 1.11412 w_k1_INTC_s106
                                 + 0.983587 w_k1_KO_s106
                                 + 0.942507 w_k1_MAT_s106
                                 + 0.85798 w_k1_MSFT_s106
                                 + 0.999624 w_k1_PG_s106 >= 1
 z_k1_s1066_constraint:          var_1 + z_k1_s1066 + 0.909335 w_k1_ADM_s106
                                 + 0.676223 w_k1_AXP_s106
                                 + 0.678283 w_k1_DIS_s106
                                 + 0.878544 w_k1_FDX_s106
                                 + 0.759561 w_k1_IBM_s106
                                 + 0.931638 w_k1_INTC_s106
                                 + 0.896809 w_k1_KO_s106
                                 + 0.839113 w_k1_MAT_s106
                                 + 0.878533 w_k1_MSFT_s106
                                 + 0.965396 w_k1_PG_s106 >= 1
 z_k1_s1067_constraint:          var_1 + z_k1_s1067 + 0.901834 w_k1_ADM_s106
                                 + 1.40222 w_k1_AXP_s106
                                 + 1.45009 w_k1_DIS_s106
                                 + 1.17804 w_k1_FDX_s106 + 1.2109 w_k1_IBM_s106
                                 + 1.39084 w_k1_INTC_s106
                                 + 1.32531 w_k1_KO_s106 + 1.26122 w_k1_MAT_s106
                                 + 1.12862 w_k1_MSFT_s106
                                 + 1.01023 w_k1_PG_s106 >= 1
 z_k1_s1070_constraint:          var_1 + z_k1_s1070 + 1.23411 w_k1_ADM_s107
                                 + 1.36879 w_k1_AXP_s107
                                 + 1.36075 w_k1_DIS_s107
                                 + 1.16663 w_k1_FDX_s107
                                 + 1.40454 w_k1_IBM_s107
                                 + 1.07469 w_k1_INTC_s107
                                 + 1.06534 w_k1_KO_s107 + 1.14137 w_k1_MAT_s107
                                 + 1.07725 w_k1_MSFT_s107
                                 + 1.13644 w_k1_PG_s107 >= 1
 z_k1_s1071_constraint:          var_1 + z_k1_s1071 + 1.06305 w_k1_ADM_s107
                                 + 0.927165 w_k1_AXP_s107
                                 + 0.78152 w_k1_DIS_s107
                                 + 0.831155 w_k1_FDX_s107
                                 + 1.17472 w_k1_IBM_s107
                                 + 0.950297 w_k1_INTC_s107
                                 + 0.90519 w_k1_KO_s107 + 1.02679 w_k1_MAT_s107
                                 + 0.809407 w_k1_MSFT_s107
                                 + 0.986504 w_k1_PG_s107 >= 1
 z_k1_s1072_constraint:          var_1 + z_k1_s1072 + 1.26397 w_k1_ADM_s107
                                 + 0.900416 w_k1_AXP_s107
                                 + 0.909262 w_k1_DIS_s107
                                 + 1.00428 w_k1_FDX_s107 + 0.8223 w_k1_IBM_s107
                                 + 1.11353 w_k1_INTC_s107
                                 + 0.841825 w_k1_KO_s107
                                 + 0.702433 w_k1_MAT_s107
                                 + 1.04785 w_k1_MSFT_s107
                                 + 0.940987 w_k1_PG_s107 >= 1
 z_k1_s1073_constraint:          var_1 + z_k1_s1073 + 1.52461 w_k1_ADM_s107
                                 + 1.09431 w_k1_AXP_s107
                                 + 1.20612 w_k1_DIS_s107
                                 + 1.20619 w_k1_FDX_s107
                                 + 0.95285 w_k1_IBM_s107
                                 + 1.2086 w_k1_INTC_s107
                                 + 0.924517 w_k1_KO_s107
                                 + 0.855599 w_k1_MAT_s107
                                 + 1.07675 w_k1_MSFT_s107
                                 + 1.03365 w_k1_PG_s107 >= 1
 z_k1_s1074_constraint:          var_1 + z_k1_s1074 + 1.5622 w_k1_ADM_s107
                                 + 1.84951 w_k1_AXP_s107
                                 + 1.91032 w_k1_DIS_s107
                                 + 1.54181 w_k1_FDX_s107
                                 + 1.17937 w_k1_IBM_s107
                                 + 1.15888 w_k1_INTC_s107
                                 + 1.22499 w_k1_KO_s107 + 1.09081 w_k1_MAT_s107
                                 + 1.24123 w_k1_MSFT_s107
                                 + 1.10998 w_k1_PG_s107 >= 1
 z_k1_s1075_constraint:          var_1 + z_k1_s1075 + 1.00412 w_k1_ADM_s107
                                 + 0.802994 w_k1_AXP_s107
                                 + 1.03025 w_k1_DIS_s107
                                 + 0.947077 w_k1_FDX_s107
                                 + 0.962617 w_k1_IBM_s107
                                 + 1.08055 w_k1_INTC_s107
                                 + 0.850663 w_k1_KO_s107
                                 + 0.491004 w_k1_MAT_s107
                                 + 0.974164 w_k1_MSFT_s107
                                 + 0.88584 w_k1_PG_s107 >= 1
 z_k1_s1076_constraint:          var_1 + z_k1_s1076 + 0.989361 w_k1_ADM_s107
                                 + 0.947729 w_k1_AXP_s107
                                 + 1.03484 w_k1_DIS_s107 + 1.1219 w_k1_FDX_s107
                                 + 0.535486 w_k1_IBM_s107
                                 + 1.20866 w_k1_INTC_s107
                                 + 0.68989 w_k1_KO_s107
                                 + 0.506692 w_k1_MAT_s107
                                 + 1.17205 w_k1_MSFT_s107
                                 + 0.911262 w_k1_PG_s107 >= 1
 z_k1_s1077_constraint:          var_1 + z_k1_s1077 + 0.88369 w_k1_ADM_s107
                                 + 0.572286 w_k1_AXP_s107
                                 + 0.697853 w_k1_DIS_s107
                                 + 0.76621 w_k1_FDX_s107
                                 + 0.730876 w_k1_IBM_s107
                                 + 0.78159 w_k1_INTC_s107
                                 + 0.696927 w_k1_KO_s107
                                 + 0.453591 w_k1_MAT_s107
                                 + 0.750242 w_k1_MSFT_s107
                                 + 0.783903 w_k1_PG_s107 >= 1
 z_k1_s1100_constraint:          var_1 + z_k1_s1100 + 1.20607 w_k1_ADM_s110
                                 + 1.52869 w_k1_AXP_s110
                                 + 1.62416 w_k1_DIS_s110
                                 + 1.35335 w_k1_FDX_s110
                                 + 1.09239 w_k1_IBM_s110
                                 + 1.10927 w_k1_INTC_s110
                                 + 1.20029 w_k1_KO_s110 + 1.60334 w_k1_MAT_s110
                                 + 1.22216 w_k1_MSFT_s110
                                 + 1.17215 w_k1_PG_s110 >= 1
 z_k1_s1101_constraint:          var_1 + z_k1_s1101 + 0.831969 w_k1_ADM_s110
                                 + 0.96764 w_k1_AXP_s110
                                 + 0.862913 w_k1_DIS_s110
                                 + 0.82812 w_k1_FDX_s110
                                 + 1.24041 w_k1_IBM_s110
                                 + 0.80385 w_k1_INTC_s110
                                 + 1.03987 w_k1_KO_s110 + 1.65371 w_k1_MAT_s110
                                 + 0.782758 w_k1_MSFT_s110
                                 + 0.946842 w_k1_PG_s110 >= 1
 z_k1_s1102_constraint:          var_1 + z_k1_s1102 + 1.17325 w_k1_ADM_s110
                                 + 0.840728 w_k1_AXP_s110
                                 + 1.13878 w_k1_DIS_s110
                                 + 1.18133 w_k1_FDX_s110
                                 + 0.698611 w_k1_IBM_s110
                                 + 1.06495 w_k1_INTC_s110
                                 + 1.00857 w_k1_KO_s110
                                 + 0.881208 w_k1_MAT_s110
                                 + 1.18043 w_k1_MSFT_s110
                                 + 1.03032 w_k1_PG_s110 >= 1
 z_k1_s1103_constraint:          var_1 + z_k1_s1103 + 0.9123 w_k1_ADM_s110
                                 + 0.769411 w_k1_AXP_s110
                                 + 0.731519 w_k1_DIS_s110
                                 + 0.863708 w_k1_FDX_s110
                                 + 0.610061 w_k1_IBM_s110
                                 + 0.889444 w_k1_INTC_s110
                                 + 0.684733 w_k1_KO_s110
                                 + 0.725036 w_k1_MAT_s110
                                 + 0.884295 w_k1_MSFT_s110
                                 + 0.808379 w_k1_PG_s110 >= 1
 z_k1_s1104_constraint:          var_1 + z_k1_s1104 + 0.850441 w_k1_ADM_s110
                                 + 0.843348 w_k1_AXP_s110
                                 + 0.615228 w_k1_DIS_s110
                                 + 0.800525 w_k1_FDX_s110
                                 + 0.793249 w_k1_IBM_s110
                                 + 0.914895 w_k1_INTC_s110
                                 + 0.786895 w_k1_KO_s110
                                 + 0.881081 w_k1_MAT_s110
                                 + 0.796837 w_k1_MSFT_s110
                                 + 0.811851 w_k1_PG_s110 >= 1
 z_k1_s1105_constraint:          var_1 + z_k1_s1105 + 1.06479 w_k1_ADM_s110
                                 + 1.13577 w_k1_AXP_s110
                                 + 0.91496 w_k1_DIS_s110
                                 + 1.00785 w_k1_FDX_s110
                                 + 0.761532 w_k1_IBM_s110
                                 + 0.902113 w_k1_INTC_s110
                                 + 0.878772 w_k1_KO_s110
                                 + 1.29693 w_k1_MAT_s110
                                 + 1.04481 w_k1_MSFT_s110
                                 + 0.876604 w_k1_PG_s110 >= 1
 z_k1_s1106_constraint:          var_1 + z_k1_s1106 + 1.06271 w_k1_ADM_s110
                                 + 1.16733 w_k1_AXP_s110
                                 + 1.19505 w_k1_DIS_s110
                                 + 1.04351 w_k1_FDX_s110
                                 + 1.09596 w_k1_IBM_s110
                                 + 1.34673 w_k1_INTC_s110
                                 + 1.21155 w_k1_KO_s110 + 1.42334 w_k1_MAT_s110
                                 + 1.18302 w_k1_MSFT_s110
                                 + 1.07059 w_k1_PG_s110 >= 1
 z_k1_s1107_constraint:          var_1 + z_k1_s1107 + 1.19666 w_k1_ADM_s110
                                 + 1.38777 w_k1_AXP_s110
                                 + 1.62165 w_k1_DIS_s110
                                 + 1.18452 w_k1_FDX_s110
                                 + 1.54668 w_k1_IBM_s110
                                 + 1.03428 w_k1_INTC_s110
                                 + 1.31891 w_k1_KO_s110 + 1.86706 w_k1_MAT_s110
                                 + 1.14856 w_k1_MSFT_s110
                                 + 1.21862 w_k1_PG_s110 >= 1
 z_k1_s1110_constraint:          var_1 + z_k1_s1110 + 0.919328 w_k1_ADM_s111
                                 + 1.01266 w_k1_AXP_s111
                                 + 1.40962 w_k1_DIS_s111 + 1.1372 w_k1_FDX_s111
                                 + 1.00369 w_k1_IBM_s111
                                 + 1.08167 w_k1_INTC_s111 + 1.2718 w_k1_KO_s111
                                 + 0.991178 w_k1_MAT_s111
                                 + 1.21135 w_k1_MSFT_s111
                                 + 1.11657 w_k1_PG_s111 >= 1
 z_k1_s1111_constraint:          var_1 + z_k1_s1111 + 1.21554 w_k1_ADM_s111
                                 + 1.15743 w_k1_AXP_s111
                                 + 1.33984 w_k1_DIS_s111
                                 + 1.38907 w_k1_FDX_s111
                                 + 0.969885 w_k1_IBM_s111
                                 + 1.30251 w_k1_INTC_s111
                                 + 1.16319 w_k1_KO_s111
                                 + 0.930722 w_k1_MAT_s111
                                 + 1.4314 w_k1_MSFT_s111 + 1.17274 w_k1_PG_s111
                                 >= 1
 z_k1_s1112_constraint:          var_1 + z_k1_s1112 + 0.919217 w_k1_ADM_s111
                                 + 0.803494 w_k1_AXP_s111
                                 + 0.928474 w_k1_DIS_s111
                                 + 0.934815 w_k1_FDX_s111
                                 + 0.930287 w_k1_IBM_s111
                                 + 0.968308 w_k1_INTC_s111
                                 + 1.05748 w_k1_KO_s111
                                 + 0.748364 w_k1_MAT_s111
                                 + 0.884176 w_k1_MSFT_s111
                                 + 0.981617 w_k1_PG_s111 >= 1
 z_k1_s1113_constraint:          var_1 + z_k1_s1113 + 1.09628 w_k1_ADM_s111
                                 + 1.14951 w_k1_AXP_s111
                                 + 1.37556 w_k1_DIS_s111
                                 + 1.06256 w_k1_FDX_s111
                                 + 1.78303 w_k1_IBM_s111
                                 + 1.17186 w_k1_INTC_s111
                                 + 1.66308 w_k1_KO_s111 + 1.47368 w_k1_MAT_s111
                                 + 1.1835 w_k1_MSFT_s111 + 1.19421 w_k1_PG_s111
                                 >= 1
 z_k1_s1114_constraint:          var_1 + z_k1_s1114 + 0.988391 w_k1_ADM_s111
                                 + 1.07883 w_k1_AXP_s111
                                 + 1.09966 w_k1_DIS_s111
                                 + 1.00972 w_k1_FDX_s111
                                 + 1.42418 w_k1_IBM_s111
                                 + 1.15926 w_k1_INTC_s111
                                 + 1.39823 w_k1_KO_s111 + 1.46278 w_k1_MAT_s111
                                 + 1.04759 w_k1_MSFT_s111
                                 + 1.08965 w_k1_PG_s111 >= 1
 z_k1_s1115_constraint:          var_1 + z_k1_s1115 + 1.06495 w_k1_ADM_s111
                                 + 0.991781 w_k1_AXP_s111
                                 + 1.02636 w_k1_DIS_s111
                                 + 1.07693 w_k1_FDX_s111
                                 + 1.19921 w_k1_IBM_s111
                                 + 1.08184 w_k1_INTC_s111
                                 + 1.21496 w_k1_KO_s111 + 1.29946 w_k1_MAT_s111
                                 + 1.11755 w_k1_MSFT_s111 + 1.0701 w_k1_PG_s111
                                 >= 1
 z_k1_s1116_constraint:          var_1 + z_k1_s1116 + 0.854129 w_k1_ADM_s111
                                 + 0.930946 w_k1_AXP_s111
                                 + 0.796244 w_k1_DIS_s111
                                 + 0.867995 w_k1_FDX_s111
                                 + 1.23063 w_k1_IBM_s111
                                 + 0.975364 w_k1_INTC_s111
                                 + 1.07178 w_k1_KO_s111
                                 + 0.990518 w_k1_MAT_s111
                                 + 0.939137 w_k1_MSFT_s111
                                 + 0.83161 w_k1_PG_s111 >= 1
 z_k1_s1117_constraint:          var_1 + z_k1_s1117 + 0.713937 w_k1_ADM_s111
                                 + 0.752007 w_k1_AXP_s111
                                 + 0.613721 w_k1_DIS_s111
                                 + 0.861648 w_k1_FDX_s111
                                 + 0.708044 w_k1_IBM_s111
                                 + 0.961151 w_k1_INTC_s111
                                 + 0.800531 w_k1_KO_s111
                                 + 0.639207 w_k1_MAT_s111
                                 + 0.821826 w_k1_MSFT_s111
                                 + 0.853615 w_k1_PG_s111 >= 1
 z_k1_s1120_constraint:          var_1 + z_k1_s1120 + 0.970235 w_k1_ADM_s112
                                 + 1.07725 w_k1_AXP_s112
                                 + 0.96549 w_k1_DIS_s112
                                 + 0.900165 w_k1_FDX_s112
                                 + 1.15215 w_k1_IBM_s112
                                 + 0.863554 w_k1_INTC_s112
                                 + 0.888918 w_k1_KO_s112
                                 + 1.18477 w_k1_MAT_s112
                                 + 0.952847 w_k1_MSFT_s112
                                 + 0.934758 w_k1_PG_s112 >= 1
 z_k1_s1121_constraint:          var_1 + z_k1_s1121 + 1.13017 w_k1_ADM_s112
                                 + 1.32535 w_k1_AXP_s112
                                 + 1.21222 w_k1_DIS_s112
                                 + 1.19145 w_k1_FDX_s112
                                 + 0.932989 w_k1_IBM_s112
                                 + 1.40718 w_k1_INTC_s112
                                 + 0.976979 w_k1_KO_s112
                                 + 0.636643 w_k1_MAT_s112
                                 + 1.1081 w_k1_MSFT_s112 + 1.05787 w_k1_PG_s112
                                 >= 1
 z_k1_s1122_constraint:          var_1 + z_k1_s1122 + 0.975229 w_k1_ADM_s112
                                 + 1.00204 w_k1_AXP_s112
                                 + 1.03677 w_k1_DIS_s112
                                 + 0.892838 w_k1_FDX_s112
                                 + 1.61609 w_k1_IBM_s112
                                 + 0.871325 w_k1_INTC_s112
                                 + 1.25135 w_k1_KO_s112 + 1.51681 w_k1_MAT_s112
                                 + 1.03358 w_k1_MSFT_s112 + 1.0261 w_k1_PG_s112
                                 >= 1
 z_k1_s1123_constraint:          var_1 + z_k1_s1123 + 1.06378 w_k1_ADM_s112
                                 + 1.55477 w_k1_AXP_s112
                                 + 1.55908 w_k1_DIS_s112
                                 + 1.23847 w_k1_FDX_s112
                                 + 1.40149 w_k1_IBM_s112
                                 + 1.57048 w_k1_INTC_s112
                                 + 1.30263 w_k1_KO_s112 + 1.71163 w_k1_MAT_s112
                                 + 1.34173 w_k1_MSFT_s112
                                 + 1.23219 w_k1_PG_s112 >= 1
 z_k1_s1124_constraint:          var_1 + z_k1_s1124 + 0.827887 w_k1_ADM_s112
                                 + 0.622809 w_k1_AXP_s112
                                 + 0.697274 w_k1_DIS_s112
                                 + 0.797188 w_k1_FDX_s112
                                 + 0.864101 w_k1_IBM_s112
                                 + 0.859746 w_k1_INTC_s112
                                 + 0.783338 w_k1_KO_s112
                                 + 0.651171 w_k1_MAT_s112
                                 + 0.797261 w_k1_MSFT_s112
                                 + 0.772303 w_k1_PG_s112 >= 1
 z_k1_s1125_constraint:          var_1 + z_k1_s1125 + 1.05948 w_k1_ADM_s112
                                 + 0.992681 w_k1_AXP_s112
                                 + 1.56176 w_k1_DIS_s112
                                 + 1.21761 w_k1_FDX_s112
                                 + 0.861289 w_k1_IBM_s112
                                 + 1.24029 w_k1_INTC_s112
                                 + 1.09168 w_k1_KO_s112
                                 + 0.743249 w_k1_MAT_s112
                                 + 1.3838 w_k1_MSFT_s112 + 1.01522 w_k1_PG_s112
                                 >= 1
 z_k1_s1126_constraint:          var_1 + z_k1_s1126 + 0.960586 w_k1_ADM_s112
                                 + 1.01102 w_k1_AXP_s112
                                 + 0.915865 w_k1_DIS_s112
                                 + 1.09636 w_k1_FDX_s112
                                 + 0.560807 w_k1_IBM_s112
                                 + 1.03888 w_k1_INTC_s112
                                 + 0.804286 w_k1_KO_s112
                                 + 0.72743 w_k1_MAT_s112
                                 + 1.01078 w_k1_MSFT_s112
                                 + 0.910634 w_k1_PG_s112 >= 1
 z_k1_s1127_constraint:          var_1 + z_k1_s1127 + 0.970365 w_k1_ADM_s112
                                 + 0.854559 w_k1_AXP_s112
                                 + 0.876217 w_k1_DIS_s112
                                 + 0.96085 w_k1_FDX_s112
                                 + 0.815722 w_k1_IBM_s112
                                 + 0.926093 w_k1_INTC_s112
                                 + 0.888636 w_k1_KO_s112
                                 + 1.01817 w_k1_MAT_s112
                                 + 0.959925 w_k1_MSFT_s112
                                 + 0.979468 w_k1_PG_s112 >= 1
 z_k1_s1130_constraint:          var_1 + z_k1_s1130 + 0.980792 w_k1_ADM_s113
                                 + 0.992873 w_k1_AXP_s113
                                 + 1.05222 w_k1_DIS_s113
                                 + 1.02608 w_k1_FDX_s113
                                 + 0.92791 w_k1_IBM_s113
                                 + 1.20368 w_k1_INTC_s113
                                 + 0.962691 w_k1_KO_s113
                                 + 0.965522 w_k1_MAT_s113
                                 + 1.13872 w_k1_MSFT_s113
                                 + 1.12598 w_k1_PG_s113 >= 1
 z_k1_s1131_constraint:          var_1 + z_k1_s1131 + 0.867478 w_k1_ADM_s113
                                 + 0.69259 w_k1_AXP_s113
                                 + 0.71525 w_k1_DIS_s113
                                 + 0.758023 w_k1_FDX_s113
                                 + 0.733207 w_k1_IBM_s113
                                 + 0.863415 w_k1_INTC_s113
                                 + 0.779523 w_k1_KO_s113
                                 + 0.672924 w_k1_MAT_s113
                                 + 0.936411 w_k1_MSFT_s113
                                 + 0.941436 w_k1_PG_s113 >= 1
 z_k1_s1132_constraint:          var_1 + z_k1_s1132 + 0.826059 w_k1_ADM_s113
                                 + 0.950509 w_k1_AXP_s113
                                 + 0.754284 w_k1_DIS_s113
                                 + 1.03371 w_k1_FDX_s113
                                 + 0.622507 w_k1_IBM_s113
                                 + 1.07283 w_k1_INTC_s113
                                 + 0.758452 w_k1_KO_s113 + 0.6148 w_k1_MAT_s113
                                 + 1.00029 w_k1_MSFT_s113
                                 + 0.961973 w_k1_PG_s113 >= 1
 z_k1_s1133_constraint:          var_1 + z_k1_s1133 + 1.15623 w_k1_ADM_s113
                                 + 1.17288 w_k1_AXP_s113
                                 + 1.43958 w_k1_DIS_s113
                                 + 1.01815 w_k1_FDX_s113
                                 + 1.77193 w_k1_IBM_s113
                                 + 1.08726 w_k1_INTC_s113
                                 + 1.39511 w_k1_KO_s113 + 1.89225 w_k1_MAT_s113
                                 + 1.40439 w_k1_MSFT_s113
                                 + 1.30126 w_k1_PG_s113 >= 1
 z_k1_s1134_constraint:          var_1 + z_k1_s1134 + 0.702863 w_k1_ADM_s113
                                 + 0.607956 w_k1_AXP_s113
                                 + 0.604553 w_k1_DIS_s113
                                 + 0.637098 w_k1_FDX_s113
                                 + 1.06645 w_k1_IBM_s113
                                 + 0.873971 w_k1_INTC_s113
                                 + 0.927634 w_k1_KO_s113
                                 + 0.883388 w_k1_MAT_s113
                                 + 0.86538 w_k1_MSFT_s113
                                 + 0.793635 w_k1_PG_s113 >= 1
 z_k1_s1135_constraint:          var_1 + z_k1_s1135 + 1.09543 w_k1_ADM_s113
                                 + 1.44043 w_k1_AXP_s113
                                 + 1.77884 w_k1_DIS_s113
                                 + 1.32867 w_k1_FDX_s113
                                 + 1.46738 w_k1_IBM_s113
                                 + 1.56689 w_k1_INTC_s113
                                 + 1.48596 w_k1_KO_s113 + 1.40136 w_k1_MAT_s113
                                 + 1.48239 w_k1_MSFT_s113
                                 + 1.26989 w_k1_PG_s113 >= 1
 z_k1_s1136_constraint:          var_1 + z_k1_s1136 + 1.59303 w_k1_ADM_s113
                                 + 1.41783 w_k1_AXP_s113 + 1.7679 w_k1_DIS_s113
                                 + 1.39262 w_k1_FDX_s113
                                 + 0.96057 w_k1_IBM_s113
                                 + 1.14429 w_k1_INTC_s113
                                 + 1.22764 w_k1_KO_s113 + 1.09564 w_k1_MAT_s113
                                 + 1.56832 w_k1_MSFT_s113
                                 + 1.32205 w_k1_PG_s113 >= 1
 z_k1_s1137_constraint:          var_1 + z_k1_s1137 + 0.939333 w_k1_ADM_s113
                                 + 0.855859 w_k1_AXP_s113
                                 + 1.06033 w_k1_DIS_s113
                                 + 0.955806 w_k1_FDX_s113
                                 + 1.24115 w_k1_IBM_s113
                                 + 1.1235 w_k1_INTC_s113 + 1.17511 w_k1_KO_s113
                                 + 1.15721 w_k1_MAT_s113
                                 + 1.15561 w_k1_MSFT_s113
                                 + 1.09902 w_k1_PG_s113 >= 1
 z_k1_s1140_constraint:          var_1 + z_k1_s1140 + 0.946338 w_k1_ADM_s114
                                 + 1.09169 w_k1_AXP_s114
                                 + 1.17554 w_k1_DIS_s114
                                 + 0.96852 w_k1_FDX_s114
                                 + 1.06202 w_k1_IBM_s114
                                 + 1.1838 w_k1_INTC_s114 + 1.18679 w_k1_KO_s114
                                 + 0.9284 w_k1_MAT_s114 + 1.0872 w_k1_MSFT_s114
                                 + 0.960749 w_k1_PG_s114 >= 1
 z_k1_s1141_constraint:          var_1 + z_k1_s1141 + 1.07653 w_k1_ADM_s114
                                 + 1.09973 w_k1_AXP_s114
                                 + 1.15241 w_k1_DIS_s114
                                 + 0.901636 w_k1_FDX_s114
                                 + 1.46987 w_k1_IBM_s114
                                 + 1.06089 w_k1_INTC_s114
                                 + 1.35858 w_k1_KO_s114 + 1.61133 w_k1_MAT_s114
                                 + 1.02783 w_k1_MSFT_s114 + 1.0393 w_k1_PG_s114
                                 >= 1
 z_k1_s1142_constraint:          var_1 + z_k1_s1142 + 1.01776 w_k1_ADM_s114
                                 + 1.50292 w_k1_AXP_s114
                                 + 1.13033 w_k1_DIS_s114
                                 + 1.13485 w_k1_FDX_s114
                                 + 1.15834 w_k1_IBM_s114
                                 + 0.984899 w_k1_INTC_s114
                                 + 1.22519 w_k1_KO_s114 + 1.55284 w_k1_MAT_s114
                                 + 1.08217 w_k1_MSFT_s114
                                 + 1.00004 w_k1_PG_s114 >= 1
 z_k1_s1143_constraint:          var_1 + z_k1_s1143 + 0.657159 w_k1_ADM_s114
                                 + 0.584173 w_k1_AXP_s114
                                 + 0.616549 w_k1_DIS_s114
                                 + 0.711528 w_k1_FDX_s114
                                 + 0.736474 w_k1_IBM_s114
                                 + 0.946297 w_k1_INTC_s114
                                 + 0.724929 w_k1_KO_s114
                                 + 0.596425 w_k1_MAT_s114
                                 + 0.785327 w_k1_MSFT_s114
                                 + 0.76596 w_k1_PG_s114 >= 1
 z_k1_s1144_constraint:          var_1 + z_k1_s1144 + 1.37106 w_k1_ADM_s114
                                 + 1.35502 w_k1_AXP_s114
                                 + 1.55358 w_k1_DIS_s114
                                 + 1.35874 w_k1_FDX_s114
                                 + 1.17333 w_k1_IBM_s114
                                 + 1.56169 w_k1_INTC_s114
                                 + 1.04983 w_k1_KO_s114
                                 + 0.799192 w_k1_MAT_s114
                                 + 1.37874 w_k1_MSFT_s114
                                 + 1.04432 w_k1_PG_s114 >= 1
 z_k1_s1145_constraint:          var_1 + z_k1_s1145 + 0.998553 w_k1_ADM_s114
                                 + 1.13206 w_k1_AXP_s114
                                 + 1.44428 w_k1_DIS_s114 + 1.2144 w_k1_FDX_s114
                                 + 0.71143 w_k1_IBM_s114
                                 + 1.31143 w_k1_INTC_s114
                                 + 1.20698 w_k1_KO_s114
                                 + 0.770817 w_k1_MAT_s114
                                 + 1.29299 w_k1_MSFT_s114 + 1.0021 w_k1_PG_s114
                                 >= 1
 z_k1_s1146_constraint:          var_1 + z_k1_s1146 + 0.920099 w_k1_ADM_s114
                                 + 0.81816 w_k1_AXP_s114
                                 + 1.07923 w_k1_DIS_s114
                                 + 1.01241 w_k1_FDX_s114
                                 + 0.745727 w_k1_IBM_s114
                                 + 1.07315 w_k1_INTC_s114
                                 + 0.896373 w_k1_KO_s114
                                 + 0.588823 w_k1_MAT_s114
                                 + 1.00988 w_k1_MSFT_s114
                                 + 0.929136 w_k1_PG_s114 >= 1
 z_k1_s1147_constraint:          var_1 + z_k1_s1147 + 0.917572 w_k1_ADM_s114
                                 + 0.871585 w_k1_AXP_s114
                                 + 0.80533 w_k1_DIS_s114
                                 + 0.858654 w_k1_FDX_s114
                                 + 0.955827 w_k1_IBM_s114
                                 + 0.876189 w_k1_INTC_s114
                                 + 0.888608 w_k1_KO_s114
                                 + 0.930663 w_k1_MAT_s114
                                 + 0.832034 w_k1_MSFT_s114
                                 + 0.910868 w_k1_PG_s114 >= 1
 z_k1_s1150_constraint:          var_1 + z_k1_s1150 + 0.85565 w_k1_ADM_s115
                                 + 1.02994 w_k1_AXP_s115
                                 + 0.85868 w_k1_DIS_s115
                                 + 0.832215 w_k1_FDX_s115
                                 + 1.17187 w_k1_IBM_s115
                                 + 0.972306 w_k1_INTC_s115
                                 + 0.925272 w_k1_KO_s115
                                 + 1.30921 w_k1_MAT_s115
                                 + 0.891493 w_k1_MSFT_s115
                                 + 0.918563 w_k1_PG_s115 >= 1
 z_k1_s1151_constraint:          var_1 + z_k1_s1151 + 0.957505 w_k1_ADM_s115
                                 + 1.1655 w_k1_AXP_s115 + 1.40197 w_k1_DIS_s115
                                 + 1.0367 w_k1_FDX_s115 + 1.23472 w_k1_IBM_s115
                                 + 1.17766 w_k1_INTC_s115
                                 + 1.21644 w_k1_KO_s115 + 1.24342 w_k1_MAT_s115
                                 + 1.18151 w_k1_MSFT_s115
                                 + 1.07717 w_k1_PG_s115 >= 1
 z_k1_s1152_constraint:          var_1 + z_k1_s1152 + 1.63487 w_k1_ADM_s115
                                 + 1.48476 w_k1_AXP_s115
                                 + 1.69623 w_k1_DIS_s115
                                 + 1.41249 w_k1_FDX_s115
                                 + 1.37754 w_k1_IBM_s115
                                 + 1.19102 w_k1_INTC_s115
                                 + 1.52297 w_k1_KO_s115 + 1.60972 w_k1_MAT_s115
                                 + 1.43422 w_k1_MSFT_s115 + 1.2693 w_k1_PG_s115
                                 >= 1
 z_k1_s1153_constraint:          var_1 + z_k1_s1153 + 1.05068 w_k1_ADM_s115
                                 + 1.16149 w_k1_AXP_s115
                                 + 1.53774 w_k1_DIS_s115 + 1.255 w_k1_FDX_s115
                                 + 0.972928 w_k1_IBM_s115
                                 + 1.05101 w_k1_INTC_s115
                                 + 1.08183 w_k1_KO_s115 + 1.49183 w_k1_MAT_s115
                                 + 1.2611 w_k1_MSFT_s115 + 1.04806 w_k1_PG_s115
                                 >= 1
 z_k1_s1154_constraint:          var_1 + z_k1_s1154 + 0.775104 w_k1_ADM_s115
                                 + 0.744758 w_k1_AXP_s115
                                 + 0.700514 w_k1_DIS_s115
                                 + 0.766624 w_k1_FDX_s115
                                 + 0.830241 w_k1_IBM_s115
                                 + 0.858219 w_k1_INTC_s115
                                 + 0.777078 w_k1_KO_s115
                                 + 0.858745 w_k1_MAT_s115
                                 + 0.712199 w_k1_MSFT_s115
                                 + 0.774568 w_k1_PG_s115 >= 1
 z_k1_s1155_constraint:          var_1 + z_k1_s1155 + 1.05493 w_k1_ADM_s115
                                 + 0.861914 w_k1_AXP_s115
                                 + 1.04684 w_k1_DIS_s115
                                 + 1.05874 w_k1_FDX_s115
                                 + 0.721644 w_k1_IBM_s115
                                 + 0.947581 w_k1_INTC_s115
                                 + 0.822619 w_k1_KO_s115
                                 + 0.77427 w_k1_MAT_s115
                                 + 1.03709 w_k1_MSFT_s115
                                 + 0.958627 w_k1_PG_s115 >= 1
 z_k1_s1156_constraint:          var_1 + z_k1_s1156 + 0.996656 w_k1_ADM_s115
                                 + 0.723689 w_k1_AXP_s115
                                 + 1.05462 w_k1_DIS_s115
                                 + 0.875641 w_k1_FDX_s115
                                 + 1.04231 w_k1_IBM_s115
                                 + 0.932442 w_k1_INTC_s115
                                 + 1.13949 w_k1_KO_s115
                                 + 0.976348 w_k1_MAT_s115
                                 + 0.975523 w_k1_MSFT_s115
                                 + 0.976582 w_k1_PG_s115 >= 1
 z_k1_s1157_constraint:          var_1 + z_k1_s1157 + 0.921758 w_k1_ADM_s115
                                 + 1.1158 w_k1_AXP_s115 + 1.17465 w_k1_DIS_s115
                                 + 0.990065 w_k1_FDX_s115
                                 + 1.68389 w_k1_IBM_s115
                                 + 1.22707 w_k1_INTC_s115
                                 + 1.45683 w_k1_KO_s115 + 2.11411 w_k1_MAT_s115
                                 + 1.11693 w_k1_MSFT_s115
                                 + 1.20487 w_k1_PG_s115 >= 1
 z_k1_s1160_constraint:          var_1 + z_k1_s1160 + 0.833475 w_k1_ADM_s116
                                 + 1.06183 w_k1_AXP_s116
                                 + 0.796035 w_k1_DIS_s116
                                 + 0.797329 w_k1_FDX_s116
                                 + 1.59464 w_k1_IBM_s116
                                 + 0.963484 w_k1_INTC_s116
                                 + 1.15527 w_k1_KO_s116 + 1.37517 w_k1_MAT_s116
                                 + 0.879125 w_k1_MSFT_s116
                                 + 1.03078 w_k1_PG_s116 >= 1
 z_k1_s1161_constraint:          var_1 + z_k1_s1161 + 1.69076 w_k1_ADM_s116
                                 + 1.52061 w_k1_AXP_s116
                                 + 1.41693 w_k1_DIS_s116
                                 + 1.38562 w_k1_FDX_s116
                                 + 0.955485 w_k1_IBM_s116
                                 + 1.24281 w_k1_INTC_s116
                                 + 1.20334 w_k1_KO_s116 + 1.43611 w_k1_MAT_s116
                                 + 1.32169 w_k1_MSFT_s116 + 1.1934 w_k1_PG_s116
                                 >= 1
 z_k1_s1162_constraint:          var_1 + z_k1_s1162 + 1.20665 w_k1_ADM_s116
                                 + 1.17695 w_k1_AXP_s116
                                 + 0.935982 w_k1_DIS_s116
                                 + 1.07288 w_k1_FDX_s116
                                 + 0.668179 w_k1_IBM_s116
                                 + 1.01529 w_k1_INTC_s116
                                 + 0.894087 w_k1_KO_s116
                                 + 0.859804 w_k1_MAT_s116
                                 + 1.00853 w_k1_MSFT_s116
                                 + 0.952347 w_k1_PG_s116 >= 1
 z_k1_s1163_constraint:          var_1 + z_k1_s1163 + 0.971935 w_k1_ADM_s116
                                 + 0.693142 w_k1_AXP_s116
                                 + 0.98777 w_k1_DIS_s116
                                 + 0.878463 w_k1_FDX_s116
                                 + 0.793159 w_k1_IBM_s116
                                 + 0.804604 w_k1_INTC_s116
                                 + 0.996461 w_k1_KO_s116
                                 + 0.692366 w_k1_MAT_s116
                                 + 1.00552 w_k1_MSFT_s116
                                 + 0.969669 w_k1_PG_s116 >= 1
 z_k1_s1164_constraint:          var_1 + z_k1_s1164 + 1.23964 w_k1_ADM_s116
                                 + 0.924832 w_k1_AXP_s116
                                 + 1.47398 w_k1_DIS_s116
                                 + 1.19999 w_k1_FDX_s116
                                 + 0.750396 w_k1_IBM_s116
                                 + 1.00377 w_k1_INTC_s116
                                 + 1.12531 w_k1_KO_s116
                                 + 0.886178 w_k1_MAT_s116
                                 + 1.2525 w_k1_MSFT_s116 + 1.03634 w_k1_PG_s116
                                 >= 1
 z_k1_s1165_constraint:          var_1 + z_k1_s1165 + 1.25339 w_k1_ADM_s116
                                 + 1.03392 w_k1_AXP_s116
                                 + 0.931875 w_k1_DIS_s116
                                 + 0.937104 w_k1_FDX_s116
                                 + 0.833112 w_k1_IBM_s116
                                 + 0.967027 w_k1_INTC_s116
                                 + 1.10788 w_k1_KO_s116 + 0.90813 w_k1_MAT_s116
                                 + 1.03252 w_k1_MSFT_s116
                                 + 1.01858 w_k1_PG_s116 >= 1
 z_k1_s1166_constraint:          var_1 + z_k1_s1166 + 0.785865 w_k1_ADM_s116
                                 + 0.787211 w_k1_AXP_s116
                                 + 0.591335 w_k1_DIS_s116
                                 + 0.656897 w_k1_FDX_s116
                                 + 0.843131 w_k1_IBM_s116
                                 + 0.733343 w_k1_INTC_s116
                                 + 0.713041 w_k1_KO_s116
                                 + 0.780499 w_k1_MAT_s116
                                 + 0.696626 w_k1_MSFT_s116
                                 + 0.778782 w_k1_PG_s116 >= 1
 z_k1_s1167_constraint:          var_1 + z_k1_s1167 + 1.27424 w_k1_ADM_s116
                                 + 1.15312 w_k1_AXP_s116
                                 + 1.24792 w_k1_DIS_s116
                                 + 0.985122 w_k1_FDX_s116
                                 + 1.26364 w_k1_IBM_s116
                                 + 1.03683 w_k1_INTC_s116 + 1.1057 w_k1_KO_s116
                                 + 1.16587 w_k1_MAT_s116
                                 + 1.0019 w_k1_MSFT_s116 + 1.03504 w_k1_PG_s116
                                 >= 1
 z_k1_s1170_constraint:          var_1 + z_k1_s1170 + 1.11151 w_k1_ADM_s117
                                 + 1.0529 w_k1_AXP_s117 + 1.30701 w_k1_DIS_s117
                                 + 1.20801 w_k1_FDX_s117
                                 + 0.820397 w_k1_IBM_s117
                                 + 1.19682 w_k1_INTC_s117 + 1.0073 w_k1_KO_s117
                                 + 0.80541 w_k1_MAT_s117
                                 + 1.30989 w_k1_MSFT_s117
                                 + 1.00802 w_k1_PG_s117 >= 1
 z_k1_s1171_constraint:          var_1 + z_k1_s1171 + 0.936329 w_k1_ADM_s117
                                 + 0.855867 w_k1_AXP_s117
                                 + 0.708598 w_k1_DIS_s117
                                 + 0.868072 w_k1_FDX_s117
                                 + 0.711124 w_k1_IBM_s117
                                 + 0.787594 w_k1_INTC_s117
                                 + 0.852399 w_k1_KO_s117
                                 + 0.647199 w_k1_MAT_s117
                                 + 0.952773 w_k1_MSFT_s117
                                 + 0.840284 w_k1_PG_s117 >= 1
 z_k1_s1172_constraint:          var_1 + z_k1_s1172 + 1.11275 w_k1_ADM_s117
                                 + 1.33108 w_k1_AXP_s117
                                 + 1.29006 w_k1_DIS_s117
                                 + 1.08385 w_k1_FDX_s117
                                 + 1.63218 w_k1_IBM_s117
                                 + 0.990401 w_k1_INTC_s117
                                 + 1.35841 w_k1_KO_s117 + 1.57584 w_k1_MAT_s117
                                 + 1.06167 w_k1_MSFT_s117
                                 + 1.04032 w_k1_PG_s117 >= 1
 z_k1_s1173_constraint:          var_1 + z_k1_s1173 + 1.00989 w_k1_ADM_s117
                                 + 0.936955 w_k1_AXP_s117
                                 + 0.908115 w_k1_DIS_s117
                                 + 0.825636 w_k1_FDX_s117
                                 + 1.3914 w_k1_IBM_s117
                                 + 0.85112 w_k1_INTC_s117
                                 + 1.19175 w_k1_KO_s117 + 1.33364 w_k1_MAT_s117
                                 + 0.953151 w_k1_MSFT_s117
                                 + 0.915983 w_k1_PG_s117 >= 1
 z_k1_s1174_constraint:          var_1 + z_k1_s1174 + 1.13161 w_k1_ADM_s117
                                 + 1.33218 w_k1_AXP_s117
                                 + 0.985684 w_k1_DIS_s117
                                 + 1.04825 w_k1_FDX_s117
                                 + 1.12607 w_k1_IBM_s117
                                 + 1.19899 w_k1_INTC_s117
                                 + 0.998139 w_k1_KO_s117
                                 + 1.45562 w_k1_MAT_s117
                                 + 0.979505 w_k1_MSFT_s117
                                 + 0.964938 w_k1_PG_s117 >= 1
 z_k1_s1175_constraint:          var_1 + z_k1_s1175 + 0.778712 w_k1_ADM_s117
                                 + 0.815415 w_k1_AXP_s117
                                 + 0.70903 w_k1_DIS_s117
                                 + 0.766658 w_k1_FDX_s117
                                 + 1.10229 w_k1_IBM_s117
                                 + 0.927784 w_k1_INTC_s117
                                 + 0.929467 w_k1_KO_s117
                                 + 0.888705 w_k1_MAT_s117
                                 + 0.853134 w_k1_MSFT_s117
                                 + 0.776056 w_k1_PG_s117 >= 1
 z_k1_s1176_constraint:          var_1 + z_k1_s1176 + 1.55011 w_k1_ADM_s117
                                 + 1.64199 w_k1_AXP_s117
                                 + 1.31425 w_k1_DIS_s117
                                 + 1.34527 w_k1_FDX_s117
                                 + 1.17072 w_k1_IBM_s117
                                 + 1.19987 w_k1_INTC_s117
                                 + 1.35753 w_k1_KO_s117 + 1.45165 w_k1_MAT_s117
                                 + 1.42108 w_k1_MSFT_s117 + 1.1488 w_k1_PG_s117
                                 >= 1
 z_k1_s1177_constraint:          var_1 + z_k1_s1177 + 0.963468 w_k1_ADM_s117
                                 + 0.821923 w_k1_AXP_s117
                                 + 0.933484 w_k1_DIS_s117
                                 + 0.871644 w_k1_FDX_s117
                                 + 0.852295 w_k1_IBM_s117
                                 + 0.979265 w_k1_INTC_s117
                                 + 1.00207 w_k1_KO_s117 + 1.07532 w_k1_MAT_s117
                                 + 1.1858 w_k1_MSFT_s117
                                 + 0.882654 w_k1_PG_s117 >= 1
 k1_return_constraint:           0.0107161859552527 w_k1_ADM_s000
                                 + 0.0405915934059159 w_k1_ADM_s001
                                 + 0.0206288933612825 w_k1_ADM_s002
                                 + 0.0349160759231503 w_k1_ADM_s003
                                 + 0.0198822846870115 w_k1_ADM_s004
                                 + 0.0177865726344152 w_k1_ADM_s005
                                 + 0.0144749045817773 w_k1_ADM_s006
                                 + 0.0169627700184612 w_k1_ADM_s007
                                 + 0.0242137948002734 w_k1_ADM_s010
                                 + 0.0598838391963086 w_k1_ADM_s011
                                 + 0.0559311779000502 w_k1_ADM_s012
                                 + 0.046154128080202 w_k1_ADM_s013
                                 + 0.0283045847928354 w_k1_ADM_s014
                                 + 0.0349444598072426 w_k1_ADM_s015
                                 + 0.0615288637298431 w_k1_ADM_s016
                                 + 0.0337468036227657 w_k1_ADM_s017
                                 + 0.0160547800647197 w_k1_ADM_s100
                                 + 0.0266141341430141 w_k1_ADM_s101
                                 + 0.00827270907362076 w_k1_ADM_s102
                                 + 0.00751366009365636 w_k1_ADM_s103
                                 + 0.0103771078250776 w_k1_ADM_s104
                                 + 0.0143932949520974 w_k1_ADM_s105
                                 + 0.00661314454092101 w_k1_ADM_s106
                                 + 0.0322270712354023 w_k1_ADM_s107
                                 + 0.0585211566655669 w_k1_ADM_s110
                                 + 0.020164398386932 w_k1_ADM_s111
                                 + 0.0683692300891352 w_k1_ADM_s112
                                 + 0.0328095489148043 w_k1_ADM_s113
                                 + 0.0777567185834897 w_k1_ADM_s114
                                 + 0.0485912967945479 w_k1_ADM_s115
                                 + 0.0317671180307847 w_k1_ADM_s116
                                 + 0.0635598843153814 w_k1_ADM_s117
                                 + 0.0128910968523837 w_k1_AXP_s000
                                 + 0.0434822028260759 w_k1_AXP_s001
                                 + 0.0168819651025855 w_k1_AXP_s002
                                 + 0.0351738659665432 w_k1_AXP_s003
                                 + 0.0221606841308431 w_k1_AXP_s004
                                 + 0.0171379977593936 w_k1_AXP_s005
                                 + 0.0139375717058051 w_k1_AXP_s006
                                 + 0.015610934917634 w_k1_AXP_s007
                                 + 0.0256298448220355 w_k1_AXP_s010
                                 + 0.0679017458250167 w_k1_AXP_s011
                                 + 0.0457961781798072 w_k1_AXP_s012
                                 + 0.0437140505476766 w_k1_AXP_s013
                                 + 0.0283801083809388 w_k1_AXP_s014
                                 + 0.0346523837042609 w_k1_AXP_s015
                                 + 0.0563049006644141 w_k1_AXP_s016
                                 + 0.0416958922256548 w_k1_AXP_s017
                                 + 0.0173731990299194 w_k1_AXP_s100
                                 + 0.0195306651757187 w_k1_AXP_s101
                                 + 0.00693403085690484 w_k1_AXP_s102
                                 + 0.0087532945578383 w_k1_AXP_s103
                                 + 0.00850575388105122 w_k1_AXP_s104
                                 + 0.0145039786998476 w_k1_AXP_s105
                                 + 0.00770978704401761 w_k1_AXP_s106
                                 + 0.0273966383415184 w_k1_AXP_s107
                                 + 0.0635160420197284 w_k1_AXP_s110
                                 + 0.019671089781268 w_k1_AXP_s111
                                 + 0.0697941941076016 w_k1_AXP_s112
                                 + 0.0312112847773682 w_k1_AXP_s113
                                 + 0.0804701210623604 w_k1_AXP_s114
                                 + 0.0479017844770924 w_k1_AXP_s115
                                 + 0.0288973367894398 w_k1_AXP_s116
                                 + 0.0632547020496386 w_k1_AXP_s117
                                 + 0.0118794359178076 w_k1_DIS_s000
                                 + 0.0406468547369628 w_k1_DIS_s001
                                 + 0.0164538076900148 w_k1_DIS_s002
                                 + 0.0340014133031143 w_k1_DIS_s003
                                 + 0.0202304331911992 w_k1_DIS_s004
                                 + 0.0177840021181204 w_k1_DIS_s005
                                 + 0.0150793250484753 w_k1_DIS_s006
                                 + 0.0152193203620525 w_k1_DIS_s007
                                 + 0.0278285338759153 w_k1_DIS_s010
                                 + 0.0628183451556401 w_k1_DIS_s011
                                 + 0.0488617190632596 w_k1_DIS_s012
                                 + 0.0401560732542054 w_k1_DIS_s013
                                 + 0.0304072543500142 w_k1_DIS_s014
                                 + 0.0312597454028376 w_k1_DIS_s015
                                 + 0.0632155582350109 w_k1_DIS_s016
                                 + 0.0375305193278113 w_k1_DIS_s017
                                 + 0.0180355421001225 w_k1_DIS_s100
                                 + 0.0213219814663049 w_k1_DIS_s101
                                 + 0.00678730976641576 w_k1_DIS_s102
                                 + 0.00792203857238661 w_k1_DIS_s103
                                 + 0.00935085346433095 w_k1_DIS_s104
                                 + 0.0139692804304295 w_k1_DIS_s105
                                 + 0.00790714035879005 w_k1_DIS_s106
                                 + 0.0287359375748265 w_k1_DIS_s107
                                 + 0.0630417564513131 w_k1_DIS_s110
                                 + 0.0212523962999507 w_k1_DIS_s111
                                 + 0.0711702888897086 w_k1_DIS_s112
                                 + 0.0353052511956658 w_k1_DIS_s113
                                 + 0.0839855008439289 w_k1_DIS_s114
                                 + 0.0552688138155714 w_k1_DIS_s115
                                 + 0.0284110373886981 w_k1_DIS_s116
                                 + 0.0621591144307504 w_k1_DIS_s117
                                 + 0.0114695046248248 w_k1_FDX_s000
                                 + 0.0414164599178748 w_k1_FDX_s001
                                 + 0.0180349572194959 w_k1_FDX_s002
                                 + 0.0340996771080745 w_k1_FDX_s003
                                 + 0.0213474697610767 w_k1_FDX_s004
                                 + 0.0168738134586223 w_k1_FDX_s005
                                 + 0.0143877655933475 w_k1_FDX_s006
                                 + 0.0150623772937896 w_k1_FDX_s007
                                 + 0.0259954505627999 w_k1_FDX_s010
                                 + 0.0626937139943506 w_k1_FDX_s011
                                 + 0.050359154852825 w_k1_FDX_s012
                                 + 0.0406116112325136 w_k1_FDX_s013
                                 + 0.029739204974301 w_k1_FDX_s014
                                 + 0.032095459423311 w_k1_FDX_s015
                                 + 0.0606902356511027 w_k1_FDX_s016
                                 + 0.0376122749323588 w_k1_FDX_s017
                                 + 0.0175341577347472 w_k1_FDX_s100
                                 + 0.0205870062409315 w_k1_FDX_s101
                                 + 0.00765088786897613 w_k1_FDX_s102
                                 + 0.0075202941608553 w_k1_FDX_s103
                                 + 0.00959895950717866 w_k1_FDX_s104
                                 + 0.0134020028328342 w_k1_FDX_s105
                                 + 0.00760621811960919 w_k1_FDX_s106
                                 + 0.028018648028662 w_k1_FDX_s107
                                 + 0.0584550198057271 w_k1_FDX_s110
                                 + 0.0215620994719585 w_k1_FDX_s111
                                 + 0.0695349535513111 w_k1_FDX_s112
                                 + 0.0318141976140118 w_k1_FDX_s113
                                 + 0.0774479670225922 w_k1_FDX_s114
                                 + 0.0491049678091984 w_k1_FDX_s115
                                 + 0.0265023509340097 w_k1_FDX_s116
                                 + 0.0609837824295738 w_k1_FDX_s117
                                 + 0.0120077353057593 w_k1_IBM_s000
                                 + 0.0415771168562794 w_k1_IBM_s001
                                 + 0.0171085356784739 w_k1_IBM_s002
                                 + 0.0347679210460413 w_k1_IBM_s003
                                 + 0.0218647798863463 w_k1_IBM_s004
                                 + 0.0175602607335284 w_k1_IBM_s005
                                 + 0.0147588229984372 w_k1_IBM_s006
                                 + 0.0157423334970551 w_k1_IBM_s007
                                 + 0.0268844080417664 w_k1_IBM_s010
                                 + 0.0617478649751087 w_k1_IBM_s011
                                 + 0.0489217967477475 w_k1_IBM_s012
                                 + 0.0415366573488295 w_k1_IBM_s013
                                 + 0.0292416791624602 w_k1_IBM_s014
                                 + 0.0336127263877293 w_k1_IBM_s015
                                 + 0.0606233843627899 w_k1_IBM_s016
                                 + 0.0393236701009665 w_k1_IBM_s017
                                 + 0.0176544261394697 w_k1_IBM_s100
                                 + 0.0201614374246592 w_k1_IBM_s101
                                 + 0.00772490742848742 w_k1_IBM_s102
                                 + 0.00785551119940454 w_k1_IBM_s103
                                 + 0.00944980992176269 w_k1_IBM_s104
                                 + 0.0133600489887398 w_k1_IBM_s105
                                 + 0.00782825119525154 w_k1_IBM_s106
                                 + 0.0273027677813338 w_k1_IBM_s107
                                 + 0.0589328595732595 w_k1_IBM_s110
                                 + 0.022302092629485 w_k1_IBM_s111
                                 + 0.0707832771448508 w_k1_IBM_s112
                                 + 0.0326033516468206 w_k1_IBM_s113
                                 + 0.0815478905230944 w_k1_IBM_s114
                                 + 0.0506075558549519 w_k1_IBM_s115
                                 + 0.026498836393291 w_k1_IBM_s116
                                 + 0.0608260347435571 w_k1_IBM_s117
                                 + 0.0118248880089466 w_k1_INTC_s000
                                 + 0.0325281925374623 w_k1_INTC_s001
                                 + 0.0162211183799431 w_k1_INTC_s002
                                 + 0.0288601966995305 w_k1_INTC_s003
                                 + 0.0228202632748599 w_k1_INTC_s004
                                 + 0.0174642592673697 w_k1_INTC_s005
                                 + 0.0153684842104499 w_k1_INTC_s006
                                 + 0.015253155163728 w_k1_INTC_s007
                                 + 0.0256542078563263 w_k1_INTC_s010
                                 + 0.0693439271414787 w_k1_INTC_s011
                                 + 0.0424490959983269 w_k1_INTC_s012
                                 + 0.0415733494738121 w_k1_INTC_s013
                                 + 0.0274179648235774 w_k1_INTC_s014
                                 + 0.0390558143908209 w_k1_INTC_s015
                                 + 0.0620388831419202 w_k1_INTC_s016
                                 + 0.0387609982325475 w_k1_INTC_s017
                                 + 0.017012679566651 w_k1_INTC_s100
                                 + 0.0205341676076695 w_k1_INTC_s101
                                 + 0.00766711178702987 w_k1_INTC_s102
                                 + 0.00794501302973825 w_k1_INTC_s103
                                 + 0.00961174194598058 w_k1_INTC_s104
                                 + 0.0124619058088906 w_k1_INTC_s105
                                 + 0.00865136867509664 w_k1_INTC_s106
                                 + 0.0286418315278822 w_k1_INTC_s107
                                 + 0.057558795377568 w_k1_INTC_s110
                                 + 0.0222893792099159 w_k1_INTC_s111
                                 + 0.0725520794831691 w_k1_INTC_s112
                                 + 0.0345137157195634 w_k1_INTC_s113
                                 + 0.0855746551109662 w_k1_INTC_s114
                                 + 0.0494588998424059 w_k1_INTC_s115
                                 + 0.0262520066536219 w_k1_INTC_s116
                                 + 0.0601990696916944 w_k1_INTC_s117
                                 + 0.0115757030407729 w_k1_KO_s000
                                 + 0.0394143586084618 w_k1_KO_s001
                                 + 0.01645653049396 w_k1_KO_s002
                                 + 0.0346006917291232 w_k1_KO_s003
                                 + 0.0209255026802666 w_k1_KO_s004
                                 + 0.0198581175019395 w_k1_KO_s005
                                 + 0.0140650167228285 w_k1_KO_s006
                                 + 0.0140862869737553 w_k1_KO_s007
                                 + 0.0268610040479143 w_k1_KO_s010
                                 + 0.0640464331697385 w_k1_KO_s011
                                 + 0.0511227381435757 w_k1_KO_s012
                                 + 0.0399532773097865 w_k1_KO_s013
                                 + 0.0277023978567488 w_k1_KO_s014
                                 + 0.0345125536561088 w_k1_KO_s015
                                 + 0.0615313776069501 w_k1_KO_s016
                                 + 0.036462763572638 w_k1_KO_s017
                                 + 0.0193485373582149 w_k1_KO_s100
                                 + 0.0208875643094863 w_k1_KO_s101
                                 + 0.00763726732709978 w_k1_KO_s102
                                 + 0.00862443532418307 w_k1_KO_s103
                                 + 0.00982579876126117 w_k1_KO_s104
                                 + 0.0141214678608447 w_k1_KO_s105
                                 + 0.00830647782804137 w_k1_KO_s106
                                 + 0.0243538900899895 w_k1_KO_s107
                                 + 0.0594718689638036 w_k1_KO_s110
                                 + 0.0237689117052345 w_k1_KO_s111
                                 + 0.0675622561668027 w_k1_KO_s112
                                 + 0.0331953213468592 w_k1_KO_s113
                                 + 0.0829234908267933 w_k1_KO_s114
                                 + 0.0507609610157974 w_k1_KO_s115
                                 + 0.0277449706045049 w_k1_KO_s116
                                 + 0.0630356669875434 w_k1_KO_s117
                                 + 0.0123041434861968 w_k1_MAT_s000
                                 + 0.0427402157406757 w_k1_MAT_s001
                                 + 0.0166107513610102 w_k1_MAT_s002
                                 + 0.0331052778677316 w_k1_MAT_s003
                                 + 0.0256811182470188 w_k1_MAT_s004
                                 + 0.0204892689827003 w_k1_MAT_s005
                                 + 0.0156986066994832 w_k1_MAT_s006
                                 + 0.0158407249947129 w_k1_MAT_s007
                                 + 0.0216572663074459 w_k1_MAT_s010
                                 + 0.0583289118570306 w_k1_MAT_s011
                                 + 0.042340193479771 w_k1_MAT_s012
                                 + 0.0444657138050191 w_k1_MAT_s013
                                 + 0.0263659762286245 w_k1_MAT_s014
                                 + 0.0385161483856701 w_k1_MAT_s015
                                 + 0.0614987116964291 w_k1_MAT_s016
                                 + 0.0449305313017111 w_k1_MAT_s017
                                 + 0.0178418293492697 w_k1_MAT_s100
                                 + 0.0153422483071482 w_k1_MAT_s101
                                 + 0.0086690155305192 w_k1_MAT_s102
                                 + 0.00901725273221209 w_k1_MAT_s103
                                 + 0.00750288296280027 w_k1_MAT_s104
                                 + 0.014576656668967 w_k1_MAT_s105
                                 + 0.00754359626705643 w_k1_MAT_s106
                                 + 0.0217019062678467 w_k1_MAT_s107
                                 + 0.0783433583000688 w_k1_MAT_s110
                                 + 0.0204077659978527 w_k1_MAT_s111
                                 + 0.0701479803332237 w_k1_MAT_s112
                                 + 0.032356360332118 w_k1_MAT_s113
                                 + 0.078752758678421 w_k1_MAT_s114
                                 + 0.0573849757476394 w_k1_MAT_s115
                                 + 0.0282227969888101 w_k1_MAT_s116
                                 + 0.0618723107485906 w_k1_MAT_s117
                                 + 0.012641738221596 w_k1_MSFT_s000
                                 + 0.0403643678760195 w_k1_MSFT_s001
                                 + 0.0165619505612363 w_k1_MSFT_s002
                                 + 0.0312075167765369 w_k1_MSFT_s003
                                 + 0.0219669193713329 w_k1_MSFT_s004
                                 + 0.0172638621389195 w_k1_MSFT_s005
                                 + 0.0152895053437048 w_k1_MSFT_s006
                                 + 0.0141839611215309 w_k1_MSFT_s007
                                 + 0.0248932424163162 w_k1_MSFT_s010
                                 + 0.0602055192682698 w_k1_MSFT_s011
                                 + 0.052113985130611 w_k1_MSFT_s012
                                 + 0.0399774241496845 w_k1_MSFT_s013
                                 + 0.0281914833016665 w_k1_MSFT_s014
                                 + 0.0331450716203663 w_k1_MSFT_s015
                                 + 0.0603683185977012 w_k1_MSFT_s016
                                 + 0.0366801524468541 w_k1_MSFT_s017
                                 + 0.018498229933436 w_k1_MSFT_s100
                                 + 0.0204056544897358 w_k1_MSFT_s101
                                 + 0.00829557856991264 w_k1_MSFT_s102
                                 + 0.00790515205609461 w_k1_MSFT_s103
                                 + 0.00881713869274159 w_k1_MSFT_s104
                                 + 0.0129540846859047 w_k1_MSFT_s105
                                 + 0.00759606408148188 w_k1_MSFT_s106
                                 + 0.0268550040975434 w_k1_MSFT_s107
                                 + 0.0583182180639188 w_k1_MSFT_s110
                                 + 0.0218841471788697 w_k1_MSFT_s111
                                 + 0.0712810869607827 w_k1_MSFT_s112
                                 + 0.0372861055562187 w_k1_MSFT_s113
                                 + 0.0804315834531957 w_k1_MSFT_s114
                                 + 0.0507362212184221 w_k1_MSFT_s115
                                 + 0.0271915426689034 w_k1_MSFT_s116
                                 + 0.0664134760808763 w_k1_MSFT_s117
                                 + 0.0119421864117177 w_k1_PG_s000
                                 + 0.0434694289042761 w_k1_PG_s001
                                 + 0.0173455665291243 w_k1_PG_s002
                                 + 0.0346569108960105 w_k1_PG_s003
                                 + 0.0221615502408385 w_k1_PG_s004
                                 + 0.0173092451250918 w_k1_PG_s005
                                 + 0.0137702660895291 w_k1_PG_s006
                                 + 0.015173559435213 w_k1_PG_s007
                                 + 0.0254262757901786 w_k1_PG_s010
                                 + 0.0588493449109446 w_k1_PG_s011
                                 + 0.0468237660792082 w_k1_PG_s012
                                 + 0.0418233708011854 w_k1_PG_s013
                                 + 0.0272973396790135 w_k1_PG_s014
                                 + 0.0332547936379389 w_k1_PG_s015
                                 + 0.0573503728400105 w_k1_PG_s016
                                 + 0.0350280770226601 w_k1_PG_s017
                                 + 0.0179821078024792 w_k1_PG_s100
                                 + 0.0205841689807563 w_k1_PG_s101
                                 + 0.0076293270898627 w_k1_PG_s102
                                 + 0.00728202585461973 w_k1_PG_s103
                                 + 0.00918684642906386 w_k1_PG_s104
                                 + 0.0131207870042748 w_k1_PG_s105
                                 + 0.00759432567647355 w_k1_PG_s106
                                 + 0.0264854704370065 w_k1_PG_s107
                                 + 0.0569811425321777 w_k1_PG_s110
                                 + 0.0213357417218589 w_k1_PG_s111
                                 + 0.0677100087006596 w_k1_PG_s112
                                 + 0.0347403897837942 w_k1_PG_s113
                                 + 0.0753820466095795 w_k1_PG_s114
                                 + 0.0483536866887391 w_k1_PG_s115
                                 + 0.0267463503909855 w_k1_PG_s116
                                 + 0.0565327918227965 w_k1_PG_s117 >= 1.15
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
