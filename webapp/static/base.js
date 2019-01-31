$(document).ready(function(){
    const $dropdownButton = $('#buses-dropdown')
    
    $dropdownButton.on('click', function(){
        $('#buses-dropdown-content').slideToggle(300)
    }).on('mouseleave', function(){
        $('#buses-dropdown-content').slideUp(300)
    })
})