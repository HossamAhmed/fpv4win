import QtQuick 2.15
import QtQuick.Controls 2.15
import realTimePlayer 1.0
import Qt.labs.platform 1.1


ApplicationWindow {
    visible: true
    width: 1024
    height: 768
    id:window
    title: qsTr("")

    QQuickRealTimePlayer {
        x: 0
        y: 0
        id: player
        width: parent.width - 200
        height:parent.height
        Component.onCompleted: {
        }
    }
    Rectangle {
        x: parent.width - 200
        y: 0
        width: 200
        height: parent.height
        color: '#cccccc'


        Column {
            padding: 5
            anchors.left: parent.left

            Rectangle {
                // Size of the background adapts to the text size plus some padding
                width: 190
                height: selDevText.height + 10
                color: "#1c80c9"

                Text {
                    id: selDevText
                    x: 5
                    anchors.verticalCenter: parent.verticalCenter
                    text: "RTL8812AU VID:PID"
                    font.pixelSize: 16
                    color: "#ffffff"
                }
            }
            ComboBox {
                id: selectDev
                width: 190
                model: ListModel {
                    id: comboBoxModel
                    Component.onCompleted: {
                        var dongleList = NativeApi.GetDongleList();
                        for (var i = 0; i < dongleList.length; i++) {
                            comboBoxModel.append({text: dongleList[i]});
                        }
                        selectDev.currentIndex = 0; // Set default selection
                    }
                }
                currentIndex: 0
            }
            Row{
                width: 190
                Column {
                    width:95
                    Rectangle {
                        // Size of the background adapts to the text size plus some padding
                        width: parent.width
                        height: selChText.height + 10
                        color: "#1c80c9"

                        Text {
                            width: parent.width
                            id: selChText
                            x: 5
                            anchors.verticalCenter: parent.verticalCenter
                            text: "Channel"
                            font.pixelSize: 16
                            color: "#ffffff"
                        }
                    }
                    ComboBox {
                        id: selectChannel
                        width: parent.width
                        model: [
                            '1','2','3','4','5','6','7','8','9','10','11','12','13',
                            '32','36','40','44','48','52','56','60','64','68','96','100','104','108','112','116','120',
                            '124','128','132','136','140','144','149','153','157','161','169','173','177'
                        ]
                        currentIndex: 39
                    }
                }
                Column {
                    width:95
                    Rectangle {
                        // Size of the background adapts to the text size plus some padding
                        width: parent.width
                        height: selCodecText.height + 10
                        color: "#1c80c9"

                        Text {
                            width: parent.width
                            id: selCodecText
                            x: 5
                            anchors.verticalCenter: parent.verticalCenter
                            text: "Codec"
                            font.pixelSize: 16
                            color: "#ffffff"
                        }
                    }
                    ComboBox {
                        id: selectCodec
                        width: parent.width
                        model: ['H264','H265']
                        currentIndex: 0
                    }
                }
            }
            Column {
                width:190
                Rectangle {
                    // Size of the background adapts to the text size plus some padding
                    width: parent.width
                    height: selBwText.height + 10
                    color: "#1c80c9"

                    Text {
                        width: parent.width
                        id: selBwText
                        x: 5
                        anchors.verticalCenter: parent.verticalCenter
                        text: "Channel Width"
                        font.pixelSize: 16
                        color: "#ffffff"
                    }
                }
                ComboBox {
                    id: selectBw
                    width: parent.width
                    model: [
                        '20',
                        '40',
                        '80',
                        '160',
                        '80_80',
                        '5',
                        '10',
                        'MAX'
                    ]
                    currentIndex: 0
                }
            }
            Rectangle {
                // Size of the background adapts to the text size plus some padding
                width: 190
                height: actionText.height + 10
                color: "#1c80c9"

                Text {
                    id: keyText
                    x: 5
                    anchors.verticalCenter: parent.verticalCenter
                    text: "Key"
                    font.pixelSize: 16
                    color: "#ffffff"
                }
            }
            Column {
                FileDialog {
                    id: fileDialog
                    title: "Select key File"
                    nameFilters: ["Key Files (*.key)"]

                    onAccepted: {
                        keySelector.text = file;
                        keySelector.text = keySelector.text.replace('file:///','')
                    }
                }
                Button {
                    width: 190
                    id:keySelector
                    text: "gs.key"
                    onClicked: fileDialog.open()
                }
            }
            Rectangle {
                // Size of the background adapts to the text size plus some padding
                width: 190
                height: actionText.height + 10
                color: "#1c80c9"

                Text {
                    id: actionText
                    x: 5
                    anchors.verticalCenter: parent.verticalCenter
                    text: "Action"
                    font.pixelSize: 16
                    color: "#ffffff"
                }
            }
            Column {
                padding:5
                Rectangle {
                    // Size of the background adapts to the text size plus some padding
                    width: 180
                    height: actionStartText.height + 10
                    color: "#2fdcf3"
                    radius: 10

                    Text {
                        id: actionStartText
                        property bool started : false;
                        x: 5
                        anchors.centerIn: parent
                        text: started?"STOP":"START"
                        font.pixelSize: 32
                        color: "#ffffff"
                    }
                    MouseArea{
                        cursorShape: Qt.PointingHandCursor
                        anchors.fill: parent
                        Component.onCompleted: {
                            NativeApi.onWifiStop.connect(()=>{
                                actionStartText.started = false;
                            });
                        }
                        onClicked: function(){
                            if(!actionStartText.started){
                                actionStartText.started = NativeApi.Start(selectDev.currentText,Number(selectChannel.currentText),Number(selectBw.currentIndex),keySelector.text);
                            }else{
                                NativeApi.Stop();
                            }
                        }
                    }
                }
            }
            Rectangle {
                // Size of the background adapts to the text size plus some padding
                width: 190
                height: countText.height + 10
                color: "#1c80c9"

                Text {
                    id: countText
                    x: 5
                    anchors.verticalCenter: parent.verticalCenter
                    text: "WFB Pkt / 802.11 Pkt"
                    font.pixelSize: 16
                    color: "#ffffff"
                }
            }
            Row {
                padding:5
                width: 190
                Text {
                    id: wfbPktCountText
                    x: 5
                    text: "0"
                    font.pixelSize: 32
                    color: "#000000"
                }
                Text {
                    x: 5
                    text: " / "
                    font.pixelSize: 32
                    color: "#000000"
                }
                Text {
                    id: airPktCountText
                    x: 5
                    text: "0"
                    font.pixelSize: 32
                    color: "#000000"
                }
            }
            Rectangle {
                id:logTitle
                z:2
                // Size of the background adapts to the text size plus some padding
                width: 190
                height: logText.height + 10
                color: "#1c80c9"

                Text {
                    id: logText
                    x: 5
                    anchors.verticalCenter: parent.verticalCenter
                    text: "WiFi Driver Log"
                    font.pixelSize: 16
                    color: "#FFFFFF"
                }
            }
            Rectangle {
                width:190
                height:window.height - 385
                color:"#f3f1f1"
                clip:true

                Component {
                    id: contactDelegate
                    Item {
                        height:log.height
                        Row {
                            padding:2
                            Text {
                                id:log
                                width: 190
                                wrapMode: Text.Wrap
                                font.pixelSize: 10
                                text: '['+level+'] '+msg
                                color: {
                                    let colors = {
                                        error: "#ff0000",
                                        info: "#0f7340",
                                        warn: "#e8c538",
                                        debug: "#3296de",
                                    }
                                    return colors[level];
                                }
                            }
                        }
                    }
                }

                ListView {
                    z:1
                    anchors.top :logTitle.bottom
                    anchors.fill: parent
                    anchors.margins:5
                    model: ListModel {}
                    delegate: contactDelegate
                    Component.onCompleted: {
                        NativeApi.onLog.connect((level,msg)=>{
                            model.append({"level": level, "msg": msg});
                            positionViewAtIndex(count - 1, ListView.End)
                        });
                    }
                }
            }
        }
    }
}